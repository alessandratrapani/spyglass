"""Test ingestion of standard ndx-pose 0.4 multi-camera 3D pose.

A minimal NWB file shaped like a DANNCE conversion: two ``CalibratedCamera``
devices (one with a ``DeviceModel``, one without), a rate-based external video
per camera, a ``MultiCameraPoseEstimation`` holding 3D series and one series-less
``PoseEstimation`` child per camera (linking the camera and its video), and a
core ``EventsTable``. The file has no task metadata.

It verifies that:

- the cameras and their calibrations are ingested
  (``CameraDevice``, ``CameraCalibration``);
- videos without task metadata get a default epoch (``TaskEpoch``, ``VideoFile``);
- the pose is ingested with its body parts and per-camera links to ``VideoFile``
  and ``CameraCalibration`` (``ImportedMultiCameraPose``), and its fetchers return
  the stored data;
- the per-camera children are not ingested as single-camera ``ImportedPose``;
- the ``EventsTable`` is ingested as one ``ImportedEvents`` entry.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from ndx_pose import (
    CalibratedCamera,
    MultiCameraPoseEstimation,
    PoseEstimation,
    PoseEstimationSeries,
    Skeleton,
    Skeletons,
)
from pynwb import NWBHDF5IO, NWBFile
from pynwb.device import DeviceModel
from pynwb.event import EventsTable
from pynwb.file import Subject
from pynwb.image import ImageSeries

N_FRAMES = 50
RATE = 50.0
CAMERAS = ("Camera1", "Camera2")
NODES = ("Snout", "TailBase")
EVENT_TIMES = (0.1, 0.4, 0.7)


@pytest.fixture(scope="module")
def multicam_nwb(verbose_context):
    from spyglass.common import Nwbfile
    from spyglass.data_import import insert_sessions
    from spyglass.settings import raw_dir
    from spyglass.utils.nwb_helper_fn import get_nwb_copy_filename

    rng = np.random.default_rng(0)
    nwbfile = NWBFile(
        session_description="test multi-camera pose",
        identifier="test_multicam_pose",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    subject = Subject(subject_id="test_multicam_subject", species="Rattus norvegicus")
    nwbfile.subject = subject

    camera_model = DeviceModel(name="test camera model", manufacturer="test maker")
    nwbfile.add_device_model(camera_model)
    cameras, videos, camera_poses = {}, {}, []
    for i, name in enumerate(CAMERAS):
        cameras[name] = nwbfile.add_device(
            CalibratedCamera(
                name=name,
                intrinsic_matrix=np.eye(3) * (i + 1),
                rotation_matrix=np.eye(3),
                translation_vector=np.array([0.0, 0.0, 10.0 * (i + 1)]),
                distortion_coefficients=np.zeros(4),
                model=camera_model if i == 0 else None,  # model is optional
            )
        )
        videos[name] = ImageSeries(
            name=f"Video{name}",
            description=f"video of {name}",
            unit="n.a.",
            format="external",
            external_file=[f"{name}.mp4"],
            starting_frame=[0],
            rate=RATE,  # rate-based: no timestamps
            num_samples=N_FRAMES,
            device=cameras[name],
        )
        nwbfile.add_acquisition(videos[name])
        camera_poses.append(
            PoseEstimation(
                name=f"{name}PoseEstimation",
                description=f"camera {name} of the pose",
                device=cameras[name],
                source_video=videos[name],
            )
        )

    skeleton = Skeleton(name="skeleton", nodes=list(NODES), edges=[[0, 1]], subject=subject)
    data = rng.random((N_FRAMES, 3))
    confidence = rng.random(N_FRAMES)
    series = [
        PoseEstimationSeries(
            name=f"PoseEstimationSeries{node}",
            description=f"{node} position",
            data=data + i,
            confidence=confidence,
            unit="millimeters",
            reference_frame="arena center",
            rate=RATE,
            starting_time=0.0,
        )
        for i, node in enumerate(NODES)
    ]
    pose = MultiCameraPoseEstimation(
        name="PoseEstimation3D",
        description="test 3D pose",
        pose_estimation_series=series,
        pose_estimations=camera_poses,
        skeleton=skeleton,
        scorer="test scorer",
        source_software="DANNCE",
        source_software_version="1.0",
    )
    behavior = nwbfile.create_processing_module("behavior", "behavior")
    behavior.add(Skeletons(skeletons=[skeleton]))
    behavior.add(pose)

    events = EventsTable(name="Contacts", description="test contacts")
    for t in EVENT_TIMES:
        events.add_row(timestamp=t)
    nwbfile.add_events_table(events)

    raw_file_name = "test_import_multicam.nwb"
    nwb_path = Path(raw_dir) / raw_file_name
    key = dict(nwb_file_name=get_nwb_copy_filename(raw_file_name))
    if (Nwbfile & key) or nwb_path.exists():
        (Nwbfile & key).delete(safemode=False)
        nwb_path.unlink(missing_ok=True)
    with NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)

    insert_sessions(raw_file_name, raise_err=True)

    yield dict(key=key, data=data, confidence=confidence)

    with verbose_context:
        (Nwbfile & key).delete(safemode=False)
        nwb_path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def multicam_pose(multicam_nwb):
    from spyglass.position.v1.imported_multicam_pose import ImportedMultiCameraPose

    return ImportedMultiCameraPose & multicam_nwb["key"]


def test_cameras_and_calibrations(multicam_nwb):
    from spyglass.common import CameraDevice
    from spyglass.position.v1.imported_multicam_pose import CameraCalibration

    cameras = (CameraDevice & [{"camera_name": n} for n in CAMERAS]).fetch(
        as_dict=True, order_by="camera_name"
    )
    assert [c["camera_id"] for c in cameras] == [1, 2]
    assert [c["model"] for c in cameras] == ["test camera model", ""]

    calibrations = CameraCalibration & multicam_nwb["key"]
    assert sorted(calibrations.fetch("camera_name")) == list(CAMERAS)
    camera2 = (calibrations & {"camera_name": "Camera2"}).fetch_calibration()
    np.testing.assert_array_equal(camera2["intrinsic_matrix"], np.eye(3) * 2)
    np.testing.assert_array_equal(camera2["translation_vector"], [0, 0, 20])


def test_default_epoch(common, multicam_nwb):
    key = multicam_nwb["key"]
    epoch = (common.TaskEpoch & key).fetch1()
    assert epoch["task_name"] == "default"
    assert epoch["interval_list_name"] == "01_default"
    assert sorted(c["camera_name"] for c in epoch["camera_names"]) == list(CAMERAS)

    valid_times = (
        common.IntervalList & key & {"interval_list_name": "01_default"}
    ).fetch1("valid_times")
    np.testing.assert_allclose(valid_times, [[0.0, (N_FRAMES - 1) / RATE]])

    videos = (common.VideoFile & key).fetch("camera_name", order_by="video_file_num")
    assert list(videos) == list(CAMERAS)


def test_multicam_pose_entries(multicam_pose):
    from spyglass.position.v1.imported_multicam_pose import ImportedMultiCameraPose

    entry = multicam_pose.fetch1()
    assert entry["pose_estimation_name"] == "PoseEstimation3D"
    assert entry["source_software"] == "DANNCE"
    assert entry["source_software_version"] == "1.0"
    assert entry["scorer"] == "test scorer"
    assert entry["interval_list_name"] == "pose_PoseEstimation3D_valid_intervals"

    parts = (ImportedMultiCameraPose.BodyPart & multicam_pose).fetch("part_name")
    assert sorted(parts) == sorted(f"PoseEstimationSeries{n}" for n in NODES)

    # Each camera links the video and the calibration of the same camera
    cameras = (ImportedMultiCameraPose.Camera & multicam_pose).fetch(
        as_dict=True, order_by="camera_name"
    )
    assert [c["camera_name"] for c in cameras] == list(CAMERAS)
    assert [c["video_file_num"] for c in cameras] == [1, 2]


def test_no_single_camera_pose(multicam_nwb):
    from spyglass.position.v1.imported_pose import ImportedPose

    # The series-less per-camera children are not single-camera poses
    assert not (ImportedPose & multicam_nwb["key"])


def test_fetch_pose_dataframe(multicam_nwb, multicam_pose):
    df = multicam_pose.fetch_pose_dataframe()
    assert df.shape == (N_FRAMES, 4 * len(NODES))
    np.testing.assert_allclose(df.index, np.arange(N_FRAMES) / RATE)
    snout = df["PoseEstimationSeriesSnout"]
    np.testing.assert_array_equal(snout[["x", "y", "z"]], multicam_nwb["data"])
    np.testing.assert_array_equal(snout["likelihood"], multicam_nwb["confidence"])


def test_fetch_skeleton_calibrations_videos(multicam_pose):
    skeleton = multicam_pose.fetch_skeleton()
    assert skeleton["nodes"] == list(NODES)
    assert skeleton["edges"] == [list(NODES)]

    calibrations = multicam_pose.fetch_calibrations()
    assert sorted(calibrations) == list(CAMERAS)
    np.testing.assert_array_equal(
        calibrations["Camera1"]["intrinsic_matrix"], np.eye(3)
    )

    assert multicam_pose.fetch_video_paths() == {
        name: f"{name}.mp4" for name in CAMERAS
    }


def test_imported_events(common, multicam_nwb):
    events = common.ImportedEvents & multicam_nwb["key"]
    entry = events.fetch1()
    assert entry["events_name"] == "Contacts"
    assert entry["n_events"] == len(EVENT_TIMES)
    np.testing.assert_allclose(
        events.fetch1_dataframe()["timestamp"], EVENT_TIMES
    )

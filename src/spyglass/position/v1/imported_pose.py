import datajoint as dj
import ndx_pose
import numpy as np
import pandas as pd
import pynwb

from spyglass.common import IntervalList, Nwbfile
from spyglass.utils.dj_mixin import SpyglassIngestion, SpyglassMixin
from spyglass.utils.nwb_helper_fn import (
    estimate_sampling_rate,
    get_valid_intervals,
)

schema = dj.schema("position_v1_imported_pose")


@schema
class ImportedPose(SpyglassIngestion, dj.Manual):
    """
    Table to ingest pose data generated prior to spyglass.
    Each entry corresponds to one ndx_pose.PoseEstimation object, or (ndx-pose
    >= 0.4.0) one MultiCameraPoseEstimation object, in an NWB file.
    PoseEstimation objects should be stored in nwb.processing.behavior
    Assumptions:
    - Single skeleton object per PoseEstimation object
    - PoseEstimation objects without series are skipped (e.g. the per-camera
      children of a MultiCameraPoseEstimation, which only link a camera and
      its source video)
    """

    _nwb_table = Nwbfile

    definition = """
    -> IntervalList
    ---
    pose_object_id: varchar(80) # unique identifier for the pose object
    skeleton_object_id: varchar(80) # unique identifier for the skeleton object
    """

    class BodyPart(SpyglassIngestion, dj.Part):
        definition = """
        -> master
        part_name: varchar(80)
        ---
        part_object_id: varchar(80)
        """

        table_key_to_obj_attr = {"self": {"part_object_id": "object_id"}}

    _source_nwb_object_type = ndx_pose.PoseEstimation

    table_key_to_obj_attr = {
        "self": {"pose_object_id": "object_id"},
        "skeleton": {"skeleton_object_id": "object_id"},
    }

    def get_nwb_objects(self, nwb_file, nwb_file_name=None):
        """Pose containers holding at least one PoseEstimationSeries.

        Includes MultiCameraPoseEstimation (ndx-pose >= 0.4.0), which is not a
        PoseEstimation subclass, and skips series-less PoseEstimation objects.
        """
        pose_types = (ndx_pose.PoseEstimation,) + tuple(
            t
            for t in (getattr(ndx_pose, "MultiCameraPoseEstimation", None),)
            if t is not None
        )
        return [
            obj
            for obj in nwb_file.objects.values()
            if isinstance(obj, pose_types) and len(obj.pose_estimation_series)
        ]

    def generate_entries_from_nwb_object(self, nwb_obj, base_key=None):
        """Generate the interval, the pose entry, and one entry per body part.

        The interval comes first: it is this table's parent, derived from the
        timestamps of the object's first body part.
        """
        nwb_file_name = base_key["nwb_file_name"]

        # use the timestamps from the first body part to define valid times
        timestamps = list(nwb_obj.pose_estimation_series.values())[
            0
        ].get_timestamps()
        sampling_rate = estimate_sampling_rate(
            timestamps, filename=nwb_file_name
        )
        valid_intervals = get_valid_intervals(
            timestamps,
            sampling_rate=sampling_rate,
            min_valid_len=sampling_rate,
            warn=not self._test_mode,
        )

        interval_pk = {
            "nwb_file_name": nwb_file_name,
            "interval_list_name": f"pose_{nwb_obj.name}_valid_intervals",
        }
        part_attr = self.BodyPart.table_key_to_obj_attr["self"]

        return {
            IntervalList: [
                {
                    **interval_pk,
                    "valid_times": valid_intervals,
                    "pipeline": "ImportedPose",
                }
            ],
            **super().generate_entries_from_nwb_object(nwb_obj, interval_pk),
            self.BodyPart: [
                dict(
                    interval_pk,
                    part_name=part,
                    **{k: getattr(part_obj, v) for k, v in part_attr.items()},
                )
                for part, part_obj in nwb_obj.pose_estimation_series.items()
            ],
        }

    def make(self, key):
        """Deprecated in favor of insert_from_nwbfile."""
        raise NotImplementedError(
            "ImportedPose.make is deprecated. Use insert_from_nwbfile."
        )

    def fetch_pose_dataframe(self, key=None):
        """Fetch pose data as a pandas DataFrame

        Parameters
        ----------
        key : dict
            Key to fetch pose data for

        Returns
        -------
        pd.DataFrame
            DataFrame containing pose data
        """
        _ = self.ensure_single_entry()
        key = key or self.fetch1("KEY")
        query = self & key
        if len(query) != 1:
            raise ValueError(f"Key selected {len(query)} entries: {query}")
        key = query.fetch1("KEY")
        pose_estimations = (
            (self & key).fetch_nwb()[0]["pose"].pose_estimation_series
        )

        index = None
        pose_df = {}
        body_parts = list(pose_estimations.keys())
        index = pose_estimations[body_parts[0]].get_timestamps()
        for body_part in body_parts:
            series = pose_estimations[body_part]
            bp_data = series.data
            part_df = {
                "video_frame_ind": np.nan,
                "x": bp_data[:, 0],
                "y": bp_data[:, 1],
            }
            if bp_data.ndim == 2 and bp_data.shape[1] == 3:  # 3D pose
                part_df["z"] = bp_data[:, 2]
            part_df["likelihood"] = (
                series.confidence[:] if series.confidence is not None else np.nan
            )

            pose_df[body_part] = pd.DataFrame(part_df, index=index)

        pose_df
        return pd.concat(pose_df, axis=1)

    def fetch_skeleton(self, key=None):
        _ = self.ensure_single_entry()
        key = key or self.fetch1("KEY")
        skeleton = (self & key).fetch_nwb()[0]["skeleton"]
        nodes = skeleton.nodes[:]
        int_edges = skeleton.edges[:]
        named_edges = [[nodes[i], nodes[j]] for i, j in int_edges]
        return {"nodes": nodes, "edges": named_edges}

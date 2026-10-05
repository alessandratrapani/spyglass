"""Ingestion of NWB core ``EventsTable`` objects (NWB >= 2.10, pynwb >= 4.0).

One row per EventsTable in the file, storing its NWB ``object_id``; the events
themselves stay in the NWB file and come back through ``fetch1_dataframe()``.
The type is matched by class name, so files without EventsTables (or older pynwb
installs without the class) are a clean no-op.
"""

import datajoint as dj

from spyglass.common.common_nwbfile import Nwbfile
from spyglass.common.common_session import Session  # noqa: F401
from spyglass.utils.dj_mixin import SpyglassIngestion

schema = dj.schema("common_events")


@schema
class ImportedEvents(SpyglassIngestion, dj.Manual):
    definition = """
    # Reference to an NWB EventsTable; the events stay in the NWB file
    -> Session
    events_name: varchar(80)          # name of the EventsTable in the NWB file
    ---
    events_object_id: varchar(40)     # NWB object id, for fetch_nwb()
    description = "": varchar(2000)
    n_events: int unsigned            # number of rows (events)
    """

    # FKs -> Session, not -> Nwbfile, so point fetch_nwb() at the file table.
    _nwb_table = Nwbfile
    _source_nwb_object_type = "EventsTable"
    # One entry per EventsTable, not one per event (row).
    _single_entry_per_table = True

    table_key_to_obj_attr = {
        "self": {
            "events_name": "name",
            "events_object_id": "object_id",
            "description": ("description", ""),
            "n_events": len,
        }
    }

    def fetch1_dataframe(self):
        """The events of one EventsTable as a DataFrame, one row per event."""
        key = self.fetch1("KEY")  # enforce exactly one row
        # fetch_nwb() already returns a DynamicTable as a DataFrame
        return (self & key).fetch_nwb()[0]["events"]

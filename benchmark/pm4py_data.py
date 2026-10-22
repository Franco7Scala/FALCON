"""Separate PM4Py input conversion: OCEL event-object relations to a typed DataFrame."""
from __future__ import annotations

from .ocel import connect_readonly


def event_object_dataframe(database, object_type: str):
    """Return one row per event-object relation for a mapped OCEL object type.

    This representation is for PM4Py's DataFrame-oriented NL query helper only.
    It deliberately preserves event IDs, activity, timestamps, object IDs/types,
    and qualifiers; it does not represent the whole relational database.
    """
    import pandas as pd
    with connect_readonly(database) as con:
        mappings = con.execute("SELECT ocel_type_map FROM event_map_type").fetchall()
        unions = " UNION ALL ".join(f'SELECT ocel_id, ocel_time FROM "event_{name}"' for (name,) in mappings)
        query = f'''WITH times AS ({unions})
            SELECT e.ocel_id AS event_id,
                   eo.ocel_object_id AS "case:concept:name",
                   COALESCE(em.ocel_type_map,e.ocel_type) AS "concept:name",
                   times.ocel_time AS "time:timestamp",
                   eo.ocel_object_id AS object_id,
                   COALESCE(om.ocel_type_map,o.ocel_type) AS object_type,
                   eo.ocel_qualifier AS qualifier
            FROM event e JOIN times ON times.ocel_id=e.ocel_id
            JOIN event_object eo ON eo.ocel_event_id=e.ocel_id
            JOIN object o ON o.ocel_id=eo.ocel_object_id
            LEFT JOIN event_map_type em ON em.ocel_type=e.ocel_type
            LEFT JOIN object_map_type om ON om.ocel_type=o.ocel_type
            WHERE COALESCE(om.ocel_type_map,o.ocel_type)=?'''
        return pd.read_sql_query(query, con, params=(object_type,))

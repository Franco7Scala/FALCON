"""Separate PM4Py input conversion: OCEL event-object relations to a typed DataFrame."""
from __future__ import annotations

from src.dataset.ocel import connect_readonly


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
        object_row = con.execute(
            "SELECT ocel_type, ocel_type_map FROM object_map_type "
            "WHERE ocel_type=? OR ocel_type_map=? LIMIT 1", (object_type, object_type)
        ).fetchone()
        if not object_row:
            raise ValueError(f"Unknown OCEL object type {object_type!r}")
        mapped_object_type = object_row[1]
        object_table = f"object_{mapped_object_type}"
        object_columns = {row[1] for row in con.execute(f'PRAGMA table_info("{object_table}")')}
        aggregates = []
        for attribute in ("weight", "price"):
            if attribute in object_columns:
                aggregates.extend((
                    f'MIN(CAST("{attribute}" AS REAL)) AS "{attribute}_min"',
                    f'MAX(CAST("{attribute}" AS REAL)) AS "{attribute}_max"',
                ))
        props_cte = (
            f'object_props AS (SELECT ocel_id, {", ".join(aggregates)} FROM "{object_table}" GROUP BY ocel_id)'
            if aggregates else "object_props AS (SELECT ocel_id FROM object)"
        )
        query = f'''WITH times AS ({unions})
            , {props_cte}, object_relations AS (
                SELECT ocel_id,
                       GROUP_CONCAT(DISTINCT source_id) AS related_source_ids,
                       GROUP_CONCAT(DISTINCT target_id) AS related_target_ids
                FROM (
                    SELECT ocel_source_id AS ocel_id, NULL AS source_id,
                           ocel_target_id AS target_id FROM object_object
                    UNION ALL
                    SELECT ocel_target_id AS ocel_id, ocel_source_id AS source_id,
                           NULL AS target_id FROM object_object
                ) rel GROUP BY ocel_id
            )
            SELECT e.ocel_id AS event_id,
                   eo.ocel_object_id AS "case:concept:name",
                   COALESCE(em.ocel_type_map,e.ocel_type) AS "concept:name",
                   times.ocel_time AS "time:timestamp",
                   eo.ocel_object_id AS object_id,
                   COALESCE(om.ocel_type_map,o.ocel_type) AS object_type,
                   eo.ocel_qualifier AS qualifier,
                   {", ".join(f'props."{attribute}"' for attribute in ("weight_min", "weight_max", "price_min", "price_max") if attribute in {item.split(' AS "')[1].split('"')[0] for item in aggregates}) or 'NULL AS weight_min, NULL AS weight_max, NULL AS price_min, NULL AS price_max'},
                   rel.related_source_ids, rel.related_target_ids
            FROM event e JOIN times ON times.ocel_id=e.ocel_id
            JOIN event_object eo ON eo.ocel_event_id=e.ocel_id
            JOIN object o ON o.ocel_id=eo.ocel_object_id
            LEFT JOIN object_props props ON props.ocel_id=eo.ocel_object_id
            LEFT JOIN object_relations rel ON rel.ocel_id=eo.ocel_object_id
            LEFT JOIN event_map_type em ON em.ocel_type=e.ocel_type
            LEFT JOIN object_map_type om ON om.ocel_type=o.ocel_type
            WHERE COALESCE(om.ocel_type_map,o.ocel_type)=?'''
        return pd.read_sql_query(query, con, params=(object_type,))

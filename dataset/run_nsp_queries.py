#!/usr/bin/env python3
"""
Esegue le 100 query NSP (foglio "NSP" di NSP_queries_cleaned.xlsx) su un database OCEL SQLite
e inserisce TUTTI i risultati ottenuti direttamente nella colonna "Expected result".

Uso:
  python run_and_save_all_expected.py --db container_logistics.sqlite --excel NSP_queries_cleaned.xlsx
"""
import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path
from openpyxl import load_workbook
import openpyxl

# ----------------------------------------------------------------------------
# Viste standard per OCEL SQLite
# ----------------------------------------------------------------------------
VIEW_EVENTI_TEMPORALI = """
CREATE TEMP VIEW IF NOT EXISTS eventi_temporali AS
SELECT
    e.ocel_id AS event_id,
    COALESCE(m.ocel_type_map, e.ocel_type) AS activity,
    t.timestamp AS timestamp
FROM event e
LEFT JOIN event_map_type m
    ON e.ocel_type = m.ocel_type
JOIN (
    SELECT ocel_id AS event_id, ocel_time AS timestamp
    FROM event_BookVehicles
    UNION ALL
    SELECT ocel_id, ocel_time FROM event_LoadTruck
    UNION ALL
    SELECT ocel_id, ocel_time FROM event_LoadtoVehicle
    UNION ALL
    SELECT ocel_id, ocel_time FROM event_DrivetoTerminal
    UNION ALL
    SELECT ocel_id, ocel_time FROM event_Weigh
    UNION ALL
    SELECT ocel_id, ocel_time FROM event_BringtoLoadingBay
    UNION ALL
    SELECT ocel_id, ocel_time FROM event_CollectGoods
    UNION ALL
    SELECT ocel_id, ocel_time FROM event_CreateTransportDocument
    UNION ALL
    SELECT ocel_id, ocel_time FROM event_Depart
    UNION ALL
    SELECT ocel_id, ocel_time FROM event_OrderEmptyContainers
    UNION ALL
    SELECT ocel_id, ocel_time FROM event_PickUpEmptyContainer
    UNION ALL
    SELECT ocel_id, ocel_time FROM event_PlaceinStock
    UNION ALL
    SELECT ocel_id, ocel_time FROM event_RegisterCustomerOrder
    UNION ALL
    SELECT ocel_id, ocel_time FROM event_RescheduleContainer
) t ON e.ocel_id = t.event_id;
"""

VIEW_EVENTI_OGGETTI = """
CREATE TEMP VIEW IF NOT EXISTS eventi_oggetti AS
SELECT
    et.event_id,
    et.activity,
    et.timestamp,
    eo.ocel_object_id AS object_id,
    COALESCE(om.ocel_type_map, o.ocel_type) AS object_type,
    eo.ocel_qualifier AS qualifier
FROM eventi_temporali et
JOIN event_object eo
    ON et.event_id = eo.ocel_event_id
JOIN object o
    ON eo.ocel_object_id = o.ocel_id
LEFT JOIN object_map_type om
    ON o.ocel_type = om.ocel_type;
"""

# Parametri automatici per le query parametrizzate
PARAMS = {
    "object_id": None,
    "container_id": None,
    "order_id": None,
    "vehicle_id": None,
    "truck_id": None,
    "document_id": None,
    "object_type": None,
    "event_id": None,
    "activity": None,
    "soglia": None,
    "status": None,
}

_TOP_OBJECT = """SELECT object_id FROM eventi_oggetti {where}
                 GROUP BY object_id ORDER BY COUNT(*) DESC, object_id LIMIT 1"""
AUTO_PARAMS = {
    "object_id": _TOP_OBJECT.format(where=""),
    "container_id": _TOP_OBJECT.format(where="WHERE object_type = 'Container'"),
    "order_id": _TOP_OBJECT.format(where="WHERE object_type = 'CustomerOrder'"),
    "vehicle_id": _TOP_OBJECT.format(where="WHERE object_type = 'Vehicle'"),
    "truck_id": _TOP_OBJECT.format(where="WHERE object_type = 'Truck'"),
    "document_id": _TOP_OBJECT.format(where="WHERE object_type = 'TransportDocument'"),
    "object_type": """SELECT object_type FROM eventi_oggetti
                      GROUP BY object_type ORDER BY COUNT(*) DESC LIMIT 1""",
    "event_id": """SELECT event_id FROM eventi_oggetti
                   GROUP BY event_id ORDER BY COUNT(*) DESC, event_id LIMIT 1""",
    "activity": """SELECT activity FROM eventi_temporali
                   GROUP BY activity ORDER BY COUNT(*) DESC LIMIT 1""",
    "soglia": "SELECT AVG(AmountofGoods) FROM object_CustomerOrder",
    "status": """SELECT Status FROM object_TransportDocument WHERE Status IS NOT NULL
                   GROUP BY Status ORDER BY COUNT(*) DESC LIMIT 1""",
}


def resolve_params(con, needed):
    params = {}
    for name in sorted(needed):
        value = PARAMS.get(name)
        if value is None and name in AUTO_PARAMS:
            try:
                row = con.execute(AUTO_PARAMS[name]).fetchone()
                value = row[0] if row else None
            except sqlite3.Error as exc:
                print(f"  [parametro :{name}] scelta automatica fallita: {exc}")
        params[name] = value
        print(f"  :{name:<13} = {value!r}")
    return params


def main():
    parser = argparse.ArgumentParser(description="Esegue le query e popola Expected result con tutti i dati nell'Excel")
    parser.add_argument("--db", required=True, help="Percorso del database .sqlite")
    parser.add_argument("--excel", default="NSP_queries_cleaned.xlsx", help="File Excel di input")
    parser.add_argument("--out", default="NSP_queries_with_all_expected.xlsx", help="File Excel di output aggiornato")
    args = parser.parse_args()

    if not Path(args.db).exists():
        sys.exit(f"Database non trovato: {args.db}")
    if not Path(args.excel).exists():
        sys.exit(f"File Excel non trovato: {args.excel}")

    # Apre il database in sola lettura
    con = sqlite3.connect(f"file:{Path(args.db).as_posix()}?mode=ro", uri=True)
    con.execute(VIEW_EVENTI_TEMPORALI)
    con.execute(VIEW_EVENTI_OGGETTI)

    # Carica il workbook con openpyxl per poter modificare le celle direttamente
    wb = load_workbook(args.excel)
    ws = wb["NSP"]

    # Individua gli indici delle colonne leggendo la prima riga
    header = [cell.value for cell in ws[1]]
    try:
        col_nsp = header.index("NSP") + 1
        col_sql = header.index("Query") + 1
        col_expected = header.index("Expected result") + 1
    except ValueError as e:
        sys.exit(f"Errore: colonna non trovata nell'intestazione del foglio NSP: {e}")

    # Raccoglie tutti i parametri necessari dalle query presenti nel foglio
    all_sql_text = " ".join([str(ws.cell(row=r, column=col_sql).value or "") for r in range(2, ws.max_row + 1)])
    needed = set(re.findall(r":([A-Za-z_]\w*)", all_sql_text))
    print("Parametri rilevati nelle query:")
    params = resolve_params(con, needed)

    ok = err = 0
    # Itera sulle righe del foglio Excel (dalla seconda in poi)
    for row_idx in range(2, ws.max_row + 1):
        nsp_val = ws.cell(row=row_idx, column=col_nsp).value
        sql_val = ws.cell(row=row_idx, column=col_sql).value

        # Salta le righe vuote o senza NSP valido
        if nsp_val is None or not str(nsp_val).strip().isdigit() or not sql_val:
            continue

        nsp_id = int(nsp_val)
        used = sorted(set(re.findall(r":([A-Za-z_]\w*)", sql_val)))
        bind = {k: params.get(k) for k in used}

        t0 = time.perf_counter()
        try:
            cur = con.execute(sql_val, bind)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            dt = time.perf_counter() - t0
            ok += 1

            # Inserisce TUTTI i risultati formattati (riga per riga separati da a capo)
            if not rows:
                result_str = "[Nessun risultato]"
            elif len(rows) == 1 and len(cols) == 1:
                result_str = str(rows[0][0])
            else:
                # Concatena tutte le righe restituite dalla query (es. le 25 righe complete)
                result_str = "\n".join(str(row) if len(row) > 1 else str(row[0]) for row in rows)

            status = "OK"
        except sqlite3.Error as exc:
            dt = time.perf_counter() - t0
            err += 1
            result_str = f"ERRORE: {exc}"
            status = "ERRORE"

        print(f"[{nsp_id}] {status} ({dt:.2f}s) -> {len(rows)} righe inserite")
        
        # Assegna il testo completo alla cella "Expected result" abilitando anche il wrap del testo
        cell = ws.cell(row=row_idx, column=col_expected)
        cell.value = result_str
        cell.alignment = openpyxl.styles.Alignment(vertical="top", wrap_text=True)

    wb.save(args.out)
    con.close()
    print(f"\nElaborazione completata: {ok} OK, {err} errori.")
    print(f"File salvato con successo in: {args.out}")


if __name__ == "__main__":
    main()
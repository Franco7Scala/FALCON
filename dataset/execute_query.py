"""
Esegue le query del benchmark OCEL su un database SQLite e scrive i risultati
nella colonna "Expected result" del file Excel.

Uso:
    python esegui_query_ocel.py --db percorso/del/database.sqlite
    python esegui_query_ocel.py --db db.sqlite --xlsx benchmark_query_ocel.xlsx --out benchmark_con_risultati.xlsx

Dipendenze: openpyxl  (pip install openpyxl)
"""
import argparse
import csv
import json
import sqlite3
from pathlib import Path

from openpyxl import load_workbook

MAX_CELL = 30000          # limite Excel: 32767 caratteri per cella
COL_TYPE, COL_QUERY, COL_EXPECTED = 1, 2, 4   # A, B, D


def serializza(v):
    """Rende un valore SQLite leggibile e serializzabile."""
    if isinstance(v, bytes):
        return v.hex()
    return v


def esegui(conn, sql):
    cur = conn.execute(sql)
    colonne = [d[0] for d in cur.description] if cur.description else []
    righe = [[serializza(v) for v in r] for r in cur.fetchall()]
    return colonne, righe


def formatta(tipo, colonne, righe, nome_csv):
    """Restituisce il testo da mettere nella cella Expected result."""
    if tipo in ("scalar", "boolean"):
        if len(righe) != 1 or len(righe[0]) != 1:
            return f"ATTENZIONE: atteso 1 valore, ottenute {len(righe)} righe"
        v = righe[0][0]
        return "NULL" if v is None else str(v)

    # tipo "set": insieme di righe
    if not righe:
        return "[]"
    testo = json.dumps(righe, ensure_ascii=False)
    if len(testo) <= MAX_CELL:
        return testo
    return (f"{len(righe)} righe (troppe per la cella) - "
            f"vedi risultati/{nome_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="file SQLite del database OCEL")
    ap.add_argument("--xlsx", default="benchmark_query_ocel.xlsx")
    ap.add_argument("--out", default="benchmark_query_ocel_risultati.xlsx")
    ap.add_argument("--cartella", default="risultati",
                    help="cartella per i risultati completi (CSV + JSON)")
    args = ap.parse_args()

    # Apertura in sola lettura: le query non possono modificare il database
    conn = sqlite3.connect(f"file:{Path(args.db).resolve()}?mode=ro", uri=True)

    wb = load_workbook(args.xlsx)
    ws = wb.active
    cartella = Path(args.cartella)
    cartella.mkdir(exist_ok=True)

    completi = {}
    ok = errori = 0

    for n, riga in enumerate(ws.iter_rows(min_row=2), start=1):
        tipo = riga[COL_TYPE - 1].value
        sql = riga[COL_QUERY - 1].value
        if not sql:
            continue
        nome_csv = f"q{n:03d}_{tipo}.csv"
        try:
            colonne, righe = esegui(conn, sql)
            testo = formatta(tipo, colonne, righe, nome_csv)
            completi[n] = {"tipo": tipo, "query": sql,
                           "colonne": colonne, "righe": righe}
            if tipo == "set":
                with open(cartella / nome_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(colonne)
                    w.writerows(righe)
            ok += 1
        except sqlite3.Error as e:
            testo = f"ERRORE: {e}"
            completi[n] = {"tipo": tipo, "query": sql, "errore": str(e)}
            errori += 1
        riga[COL_EXPECTED - 1].value = testo

    with open(cartella / "risultati_completi.json", "w", encoding="utf-8") as f:
        json.dump(completi, f, ensure_ascii=False, indent=1)

    wb.save(args.out)
    conn.close()
    print(f"Query eseguite: {ok} ok, {errori} con errore")
    print(f"Excel aggiornato: {args.out}")
    print(f"Risultati completi in: {cartella}/")


if __name__ == "__main__":
    main()

# C:\Users\domen\Documents\GitHub\AgenticQueringOCEL\databases\container_logistics.sqlite
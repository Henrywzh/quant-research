from typing import Iterable
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from db.models import MacroSeries, MacroObservation

def get_or_create_series(
    s: Session, *,
    code: str,
    country: str,
    name: str | None = None,
    provider: str | None = None,
    provider_code: str | None = None,
    frequency: str | None = None,
    unit: str | None = None,
    category: str | None = None,
) -> MacroSeries:
    country = country.upper()
    row = s.execute(
        select(MacroSeries).where(MacroSeries.code == code, MacroSeries.country == country)
    ).scalar_one_or_none()

    if row:
        changed = False
        for k, v in dict(name=name, provider=provider, provider_code=provider_code,
                         frequency=frequency, unit=unit, category=category).items():
            if v is not None and getattr(row, k) != v:
                setattr(row, k, v); changed = True
        if changed:
            s.add(row)
        return row

    obj = MacroSeries(
        code=code, name=name, provider=provider, provider_code=provider_code,
        frequency=frequency, unit=unit, country=country, category=category
    )
    s.add(obj)
    s.flush()

    return obj

def upsert_observations(s: Session, *, macro_id: int, rows: Iterable[tuple]):
    payload = []
    for r in rows:
        if len(r) == 2:
            ts, value = r; rev = 0
        else:
            ts, value, rev = r
        payload.append({"macro_id": macro_id, "ts": ts, "value": value, "revision": rev})

    if not payload:
        return 0

    stmt = insert(MacroObservation.__table__).values(payload)
    stmt = stmt.on_conflict_do_update(
        index_elements=["macro_id", "ts", "revision"],
        set_={"value": stmt.excluded.value}
    )
    res = s.execute(stmt)
    return res.rowcount or 0

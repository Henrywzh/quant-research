from __future__ import annotations
from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, Float, Text, TIMESTAMP, ForeignKey, CheckConstraint, UniqueConstraint

class Base(DeclarativeBase):
    pass

class MacroSeries(Base):
    __tablename__ = "macro_series"
    macro_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String, nullable=False)        # e.g. DFF, CPIAUCSL
    name: Mapped[str | None] = mapped_column(Text)
    provider: Mapped[str | None] = mapped_column(String)             # FRED, ECB, ONS, etc.
    provider_code: Mapped[str | None] = mapped_column(String)
    frequency: Mapped[str | None] = mapped_column(String)            # daily/weekly/monthly/quarterly/annual
    unit: Mapped[str | None] = mapped_column(String)
    country: Mapped[str | None] = mapped_column(String)              # 'US','EA19','CN'
    category: Mapped[str | None] = mapped_column(String)

    observations: Mapped[list["MacroObservation"]] = relationship(back_populates="series")

    __table_args__ = (
        UniqueConstraint("code", "country", name="uq_macro_series_code_country"),
        CheckConstraint(
            "frequency IN ('daily','weekly','monthly','quarterly','annual')",
            name="chk_macro_freq"
        ),
    )

class MacroObservation(Base):
    __tablename__ = "macro_observations"
    macro_id: Mapped[int] = mapped_column(ForeignKey("macro_series.macro_id", ondelete="CASCADE"), primary_key=True)
    ts: Mapped["datetime"] = mapped_column(TIMESTAMP, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, primary_key=True, default=0)
    value: Mapped[float | None] = mapped_column(Float)
    load_id: Mapped[str | None] = mapped_column(String)

    series: Mapped[MacroSeries] = relationship(back_populates="observations")

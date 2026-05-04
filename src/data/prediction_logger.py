"""
Persists every /predict call to a SQLite database.

The stored rows are the raw material for DriftDetector.build_current_data().

Schema
------
predictions
  id            INTEGER  PRIMARY KEY AUTOINCREMENT
  timestamp     DATETIME UTC
  sepal_length  REAL
  sepal_width   REAL
  petal_length  REAL
  petal_width   REAL
  prediction    TEXT     (class name)
  class_id      INTEGER
  confidence    REAL
  model_source  TEXT
  latency_ms    REAL
"""

import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Session

from src.config import settings


class _Base(DeclarativeBase):
    pass


class PredictionRecord(_Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    prediction = Column(String, nullable=False)
    class_id = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    model_source = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=True)


class PredictionLogger:
    """Thread-safe SQLAlchemy-based prediction store."""

    def __init__(self, db_url: str = settings.predictions_db_url):
        # Ensure parent directory exists for SQLite file
        db_path = db_url.replace("sqlite:///", "")
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self._engine = create_engine(db_url, connect_args={"check_same_thread": False})
        _Base.metadata.create_all(self._engine)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log(
        self,
        features: dict,
        prediction: str,
        class_id: int,
        confidence: float,
        model_source: str,
        latency_ms: float | None = None,
    ) -> None:
        """Persist one prediction row. Non-blocking — failures are silently ignored."""
        try:
            with Session(self._engine) as session:
                session.add(
                    PredictionRecord(
                        sepal_length=features["sepal_length"],
                        sepal_width=features["sepal_width"],
                        petal_length=features["petal_length"],
                        petal_width=features["petal_width"],
                        prediction=prediction,
                        class_id=class_id,
                        confidence=confidence,
                        model_source=model_source,
                        latency_ms=latency_ms,
                    )
                )
                session.commit()
        except Exception:
            pass  # never crash the API because of a logging failure

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_recent(self, limit: int = 1000) -> list[dict]:
        """Return the N most recent predictions as plain dicts (for drift detection)."""
        with Session(self._engine) as session:
            rows = (
                session.query(PredictionRecord)
                .order_by(PredictionRecord.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "sepal_length": r.sepal_length,
                    "sepal_width": r.sepal_width,
                    "petal_length": r.petal_length,
                    "petal_width": r.petal_width,
                    "class_id": r.class_id,
                    "prediction": r.prediction,
                    "confidence": r.confidence,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in rows
            ]

    def stats(self) -> dict:
        """Return aggregate statistics useful for the /stats API endpoint."""
        with Session(self._engine) as session:
            total = session.query(func.count(PredictionRecord.id)).scalar()
            avg_conf = session.query(func.avg(PredictionRecord.confidence)).scalar()
            avg_latency = session.query(func.avg(PredictionRecord.latency_ms)).scalar()

            class_counts: dict[str, int] = {}
            for row in (
                session.query(
                    PredictionRecord.prediction,
                    func.count(PredictionRecord.id),
                )
                .group_by(PredictionRecord.prediction)
                .all()
            ):
                class_counts[row[0]] = row[1]

            return {
                "total_predictions": total or 0,
                "avg_confidence": round(avg_conf, 4) if avg_conf else None,
                "avg_latency_ms": round(avg_latency, 2) if avg_latency else None,
                "class_distribution": class_counts,
            }

    def count(self) -> int:
        with Session(self._engine) as session:
            return session.query(func.count(PredictionRecord.id)).scalar() or 0

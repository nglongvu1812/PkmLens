"""Shared SQLAlchemy schema. Only core-api writes business state."""
import os
from pathlib import Path
from sqlalchemy import event, create_engine, String, Text, Integer, JSON, UniqueConstraint, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATA = Path(os.getenv("DATA_DIR", "data")).resolve()
DATA.mkdir(parents=True, exist_ok=True)
URL = os.getenv("DATABASE_URL", "sqlite:///" + (DATA / "pkmlens.db").as_posix())
engine = create_engine(URL, connect_args={"check_same_thread": False} if URL.startswith("sqlite") else {}, pool_pre_ping=True)
if URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def sqlite_foreign_keys(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")
Session = sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class Card(Base):
    __tablename__ = "cards"
    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    provider_id: Mapped[str] = mapped_column(String(100), unique=True)
    set_id: Mapped[str] = mapped_column(String(80), index=True)
    number: Mapped[str] = mapped_column(String(40))
    language: Mapped[str] = mapped_column(String(10), default="en")
    name: Mapped[str] = mapped_column(String(200), index=True)
    image: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[dict] = mapped_column(JSON)

class Mapping(Base):
    __tablename__ = "provider_mappings"
    __table_args__ = (UniqueConstraint("provider", "provider_id", "variant", "card_id", name="uq_provider_mapping_identity"), Index("ix_provider_mappings_provider_id", "provider_id"))
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    card_id: Mapped[str | None] = mapped_column(ForeignKey("cards.id"), nullable=True)
    provider: Mapped[str] = mapped_column(String(40))
    provider_id: Mapped[str] = mapped_column(String(100))
    variant: Mapped[str] = mapped_column(String(180))
    status: Mapped[str] = mapped_column(String(30))

class Price(Base):
    __tablename__ = "price_snapshots"
    __table_args__ = (UniqueConstraint("card_id", "variant", "source", "source_at"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"), index=True)
    variant: Mapped[str] = mapped_column(String(180))
    source: Mapped[str] = mapped_column(String(80))
    source_at: Mapped[str] = mapped_column(String(50))
    fetched_at: Mapped[str] = mapped_column(String(50))
    currency: Mapped[str] = mapped_column(String(3))
    amount: Mapped[str] = mapped_column(String(40))
    low: Mapped[str | None] = mapped_column(String(40), nullable=True)
    high: Mapped[str | None] = mapped_column(String(40), nullable=True)
    evidence: Mapped[dict] = mapped_column(JSON)

class Fx(Base):
    __tablename__ = "fx_snapshots"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_at: Mapped[str] = mapped_column(String(80))
    fetched_at: Mapped[str] = mapped_column(String(50))
    source: Mapped[str] = mapped_column(Text)
    rates: Mapped[dict] = mapped_column(JSON)

class Identity(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(50))
    expires_at: Mapped[str] = mapped_column(String(50))

class Scan(Base):
    __tablename__ = "scans"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    created_at: Mapped[str] = mapped_column(String(50))
    result: Mapped[dict] = mapped_column(JSON)
    card_id: Mapped[str | None] = mapped_column(ForeignKey("cards.id"), nullable=True)
    variant: Mapped[str | None] = mapped_column(String(180), nullable=True)
    condition: Mapped[str] = mapped_column(String(10), default="NM")
    revision: Mapped[int] = mapped_column(Integer, default=0)

class Quote(Base):
    __tablename__ = "quotes"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id"), index=True)
    owner: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    revision: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict] = mapped_column(JSON)

class Listing(Base):
    __tablename__ = "listings"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id"), unique=True)
    quote_id: Mapped[str] = mapped_column(ForeignKey("quotes.id"))
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"))
    variant: Mapped[str] = mapped_column(String(180))
    condition: Mapped[str] = mapped_column(String(10))
    asking_vnd: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="draft")
    created_at: Mapped[str] = mapped_column(String(50))
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True)
    password: Mapped[str] = mapped_column(Text)
    owner: Mapped[str] = mapped_column(ForeignKey("sessions.id"), unique=True)
    display_name: Mapped[str] = mapped_column(String(80))

class Login(Base):
    __tablename__ = "logins"
    id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))

class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (UniqueConstraint("buyer", "request_key"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    buyer: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    request_key: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(30))
    total_vnd: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(String(50))
    details: Mapped[dict] = mapped_column(JSON)

class Ledger(Base):
    __tablename__ = "wallet_ledger"
    __table_args__ = (UniqueConstraint("owner", "reference"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    delta_vnd: Mapped[int] = mapped_column(Integer)
    reference: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[str] = mapped_column(String(50))

class PersonalCard(Base):
    __tablename__ = "personal_cards"
    __table_args__ = (UniqueConstraint("owner", "card_id", "variant", "kind"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"))
    variant: Mapped[str] = mapped_column(String(180))
    kind: Mapped[str] = mapped_column(String(20))
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    threshold_vnd: Mapped[int | None] = mapped_column(Integer, nullable=True)
    alert_state: Mapped[str] = mapped_column(String(20), default="armed")

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    listing_id: Mapped[str] = mapped_column(ForeignKey("listings.id"), index=True)
    buyer: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    sender: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(50))


class LoginAttempt(Base):
    __tablename__ = "login_attempts"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    window_started: Mapped[str] = mapped_column(String(50))
    blocked_until: Mapped[str | None] = mapped_column(String(50), nullable=True)

class PriceAlert(Base):
    __tablename__ = "price_alerts"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    card_id: Mapped[str] = mapped_column(ForeignKey("cards.id"))
    variant: Mapped[str] = mapped_column(String(180))
    direction: Mapped[str] = mapped_column(String(10))
    threshold_vnd: Mapped[int] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(default=True)
    triggered_at: Mapped[str | None] = mapped_column(String(50), nullable=True)

class Notification(Base):
    __tablename__ = "notifications"
    __table_args__ = (UniqueConstraint("owner", "alert_id"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    alert_id: Mapped[str] = mapped_column(ForeignKey("price_alerts.id"))
    created_at: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict] = mapped_column(JSON)


class RoleGrant(Base):
    __tablename__ = "role_grants"
    owner: Mapped[str] = mapped_column(ForeignKey("sessions.id"), primary_key=True)
    role: Mapped[str] = mapped_column(String(20))

class Audit(Base):
    __tablename__ = "audit_logs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    actor: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    action: Mapped[str] = mapped_column(String(80))
    target: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[str] = mapped_column(String(50))
    details: Mapped[dict] = mapped_column(JSON)

class CollectionValue(Base):
    __tablename__ = "collection_values"
    __table_args__ = (UniqueConstraint("owner", "day"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    day: Mapped[str] = mapped_column(String(10))
    amount_vnd: Mapped[str] = mapped_column(String(40))
    missing_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[str] = mapped_column(String(50))

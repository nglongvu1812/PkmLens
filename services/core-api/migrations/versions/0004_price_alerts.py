"""Persistent price alerts and one-shot local notifications."""
from alembic import op
from cardlens_core.db import Base
revision="0004"
down_revision="0003"
branch_labels=None
depends_on=None

def upgrade():
    for name in ("price_alerts","notifications"):
        Base.metadata.tables[name].create(op.get_bind(),checkfirst=True)

def downgrade():
    for name in ("notifications","price_alerts"):
        Base.metadata.tables[name].drop(op.get_bind(),checkfirst=True)

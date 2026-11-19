"""Roles, moderation audit and measured collection-value snapshots."""
from alembic import op
from cardlens_core.db import Base
revision="0006"
down_revision="0005"
branch_labels=None
depends_on=None

def upgrade():
    for name in ("role_grants","audit_logs","collection_values"):
        Base.metadata.tables[name].create(op.get_bind(),checkfirst=True)

def downgrade():
    for name in ("collection_values","audit_logs","role_grants"):
        Base.metadata.tables[name].drop(op.get_bind(),checkfirst=True)

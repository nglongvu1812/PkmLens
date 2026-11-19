"""Persist login throttling across worker restarts."""
from alembic import op
from cardlens_core.db import Base
revision="0003"
down_revision="0002"
branch_labels=None
depends_on=None

def upgrade():
    Base.metadata.tables["login_attempts"].create(op.get_bind(),checkfirst=True)

def downgrade():
    Base.metadata.tables["login_attempts"].drop(op.get_bind(),checkfirst=True)


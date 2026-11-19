"""Initial catalog, snapshots, scan ownership, quote and listing schema."""
from alembic import op
from cardlens_core.db import Base
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    
    for name in ("cards","provider_mappings","price_snapshots","fx_snapshots","sessions","scans","quotes","listings"):
        Base.metadata.tables[name].create(op.get_bind(), checkfirst=True)

def downgrade():
    
    for name in ("listings","quotes","scans","sessions","fx_snapshots","price_snapshots","provider_mappings","cards"):
        Base.metadata.tables[name].drop(op.get_bind(), checkfirst=True)


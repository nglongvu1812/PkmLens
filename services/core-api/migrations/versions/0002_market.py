"""Accounts, order/escrow ledger, collection/wishlist and private chat."""
from alembic import op
from cardlens_core.db import Base
revision="0002"
down_revision="0001"
branch_labels=None
depends_on=None

def upgrade():
    for name in ("users","logins","orders","wallet_ledger","personal_cards","messages"):
        Base.metadata.tables[name].create(op.get_bind(), checkfirst=True)

def downgrade():
    for name in ("messages","personal_cards","wallet_ledger","orders","logins","users"):
        Base.metadata.tables[name].drop(op.get_bind(), checkfirst=True)


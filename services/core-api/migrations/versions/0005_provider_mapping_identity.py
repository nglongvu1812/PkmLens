"""Allow a provider product ID to map to multiple card identities."""
from alembic import op
from sqlalchemy import text, inspect
revision="0005"
down_revision="0004"
branch_labels=None
depends_on=None

def upgrade():
    bind=op.get_bind()
    if bind.dialect.name=="sqlite":
        bind.execute(text("CREATE TABLE provider_mappings_new (id VARCHAR(64) PRIMARY KEY, card_id VARCHAR(160) REFERENCES cards(id), provider VARCHAR(40) NOT NULL, provider_id VARCHAR(100) NOT NULL, variant VARCHAR(180) NOT NULL, status VARCHAR(30) NOT NULL, CONSTRAINT uq_provider_mapping_identity UNIQUE (provider, provider_id, variant, card_id))"))
        bind.execute(text("INSERT INTO provider_mappings_new SELECT id,card_id,provider,provider_id,variant,status FROM provider_mappings"))
        bind.execute(text("DROP TABLE provider_mappings"))
        bind.execute(text("ALTER TABLE provider_mappings_new RENAME TO provider_mappings"))
        bind.execute(text("CREATE INDEX ix_provider_mappings_provider_id ON provider_mappings (provider_id)"))
    else:
        constraints=inspect(bind).get_unique_constraints("provider_mappings")
        for constraint in constraints:
            if set(constraint["column_names"])=={"provider","provider_id","variant"}:
                op.drop_constraint(constraint["name"],"provider_mappings",type_="unique")
        if not any(set(c["column_names"])=={"provider","provider_id","variant","card_id"} for c in constraints):
            op.create_unique_constraint("uq_provider_mapping_identity","provider_mappings",["provider","provider_id","variant","card_id"])

def downgrade():
    bind=op.get_bind()
    if bind.dialect.name=="sqlite":
        duplicates=bind.execute(text("SELECT 1 FROM provider_mappings GROUP BY provider,provider_id,variant HAVING COUNT(*)>1 LIMIT 1")).first()
        if duplicates: raise RuntimeError("Cannot downgrade: ambiguous provider IDs exist across cards")
        bind.execute(text("CREATE TABLE provider_mappings_old (id VARCHAR(64) PRIMARY KEY, card_id VARCHAR(160) REFERENCES cards(id), provider VARCHAR(40) NOT NULL, provider_id VARCHAR(100) NOT NULL, variant VARCHAR(180) NOT NULL, status VARCHAR(30) NOT NULL, UNIQUE(provider,provider_id,variant))"))
        bind.execute(text("INSERT INTO provider_mappings_old SELECT id,card_id,provider,provider_id,variant,status FROM provider_mappings"))
        bind.execute(text("DROP TABLE provider_mappings")); bind.execute(text("ALTER TABLE provider_mappings_old RENAME TO provider_mappings"))
    else:
        op.drop_constraint("uq_provider_mapping_identity","provider_mappings",type_="unique")
        op.create_unique_constraint("provider_mappings_provider_provider_id_variant_key","provider_mappings",["provider","provider_id","variant"])

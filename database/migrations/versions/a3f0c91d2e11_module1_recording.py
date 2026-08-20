"""module 1: translation pref + lecture recordings

Revision ID: a3f0c91d2e11
Revises: 1c40ff411020
Create Date: 2026-08-07 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'a3f0c91d2e11'
down_revision = '1c40ff411020'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('sessions', sa.Column('target_lang', sa.String(length=16), nullable=False, server_default='hi'))

    op.create_table(
        'recordings',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('language', sa.String(length=16), nullable=False),
        sa.Column('duration_s', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('file_path', sa.String(length=512), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('recordings')
    op.drop_column('sessions', 'target_lang')
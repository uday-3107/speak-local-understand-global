"""add sessions.join_code

Revision ID: b7e3f9a1c2d4
Revises: a3f0c91d2e11
Create Date: 2026-08-10 12:30:00.000000

"""
import uuid

from alembic import op
import sqlalchemy as sa


revision = 'b7e3f9a1c2d4'
down_revision = 'a3f0c91d2e11'
branch_labels = None
depends_on = None

ALPHABET = "023456789ABCDEFGHJKMNPQRSTUVWXYZ"


def code_for(session_id) -> str:
    value = uuid.UUID(str(session_id)).int & ((1 << 30) - 1)
    code = ""
    for _ in range(6):
        code = ALPHABET[value % 32] + code
        value //= 32
    return code


def upgrade() -> None:
    op.add_column('sessions', sa.Column('join_code', sa.String(length=8), nullable=True))
    conn = op.get_bind()
    rows = conn.execute(sa.text("SELECT id FROM sessions WHERE join_code IS NULL")).fetchall()
    for (session_id,) in rows:
        conn.execute(
            sa.text("UPDATE sessions SET join_code = :code WHERE id = :sid"),
            {"code": code_for(session_id), "sid": session_id},
        )
    op.create_unique_constraint('uq_sessions_join_code', 'sessions', ['join_code'])


def downgrade() -> None:
    op.drop_constraint('uq_sessions_join_code', 'sessions', type_='unique')
    op.drop_column('sessions', 'join_code')
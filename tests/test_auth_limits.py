from fastapi.testclient import TestClient
from fastapi import HTTPException
from sqlalchemy import select
from cardlens_core.main import app
from cardlens_core.db import Session,LoginAttempt
from cardlens_core.accounts import email_key,password_hash

def test_login_rate_limit_persists_and_hashes_email():
    client=TestClient(app)
    client.post("/api/v1/session")
    for _ in range(8):
        response=client.post("/api/v1/auth/login",json={"email":"Nobody@Example.com","password":"wrong-password-1234"})
        assert response.status_code==401
    response=client.post("/api/v1/auth/login",json={"email":"nobody@example.com","password":"wrong-password-1234"})
    assert response.status_code==429
    with Session() as db:
        attempts=db.scalars(select(LoginAttempt)).all()
        assert len(attempts)==1 and attempts[0].failed_count==8
    assert email_key(" Alice@Example.com ")=="alice@example.com"
    assert password_hash("correct-horse-battery")!=password_hash("correct-horse-battery")

def test_logout_old_guest_token_cannot_become_a_new_owner():
    client=TestClient(app)
    client.post("/api/v1/session")
    token=client.cookies.get("pkmlens_session")
    assert client.get("/api/v1/me").status_code==200
    assert client.post("/api/v1/auth/logout").status_code==200
    assert client.get("/api/v1/listings/drafts").status_code==401
    client.cookies.set("pkmlens_session",token)
    assert client.get("/api/v1/listings/drafts").status_code==401



def test_expired_login_attempt_window_resets_existing_row():
    from datetime import datetime,timedelta,timezone
    c=TestClient(app)
    c.post("/api/v1/auth/login",json={"email":"reset@example.com","password":"wrong-password-1234"})
    with Session.begin() as db:
        row=db.scalar(select(LoginAttempt))
        row.window_started=(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()
        row.blocked_until=None
    r=c.post("/api/v1/auth/login",json={"email":"reset@example.com","password":"wrong-password-1234"})
    assert r.status_code==401
    with Session() as db:
        rows=db.scalars(select(LoginAttempt)).all()
        assert len(rows)==1 and rows[0].failed_count==1


def test_registration_rotates_and_revokes_guest_session():
    c=TestClient(app)
    c.post("/api/v1/session")
    old=c.cookies.get("pkmlens_session")
    r=c.post("/api/v1/auth/register",json={"email":"rotate@example.com","password":"correct-password-1234"})
    assert r.status_code==200
    assert old!=c.cookies.get("pkmlens_session")
    stolen=TestClient(app)
    stolen.cookies.set("pkmlens_session",old)
    assert stolen.get("/api/v1/me").status_code==401
    assert c.get("/api/v1/me").json()["registered"] is True

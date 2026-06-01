"""Schema creation and seed runner. Safe to run multiple times."""
import sys
import pandas as pd
from app_config import settings
from db import Base, Role, User, engine, session_scope
from repositories import AuditLogRepository, SalesRepository, UserRepository
from services.auth_service import _hash_password


def create_schema():
    print(f"[migrate] Creating schema at {settings.db_path}")
    Base.metadata.create_all(bind=engine)


def seed_admin():
    with session_scope() as s:
        users = UserRepository(s)
        if users.get_by_email(settings.default_admin_email):
            print(f"[seed] Admin {settings.default_admin_email} already exists")
            return
        admin = User(
            email=settings.default_admin_email,
            password_hash=_hash_password(settings.default_admin_password),
            role=Role.ADMIN,
            must_change_password=True,
        )
        users.add(admin)
        AuditLogRepository(s).log(action="seed.admin_created", details={"email": admin.email})
        print(f"[seed] Created admin {admin.email} / {settings.default_admin_password}")


def seed_sales(force: bool = False):
    csv = settings.csv_seed_path
    if not csv.exists():
        print(f"[seed] CSV not found at {csv}; skipping")
        return
    with session_scope() as s:
        sales = SalesRepository(s)
        existing = sales.count()
        if existing > 0 and not force:
            print(f"[seed] sales_records already has {existing:,} rows; skipping (use --force to reseed)")
            return
        if force:
            print("[seed] --force: truncating sales_records")
            sales.truncate()
        df = pd.read_csv(csv)
        df["date"] = pd.to_datetime(df["date"])
        rows = df.to_dict(orient="records")
        n = sales.bulk_insert(rows)
        AuditLogRepository(s).log(action="seed.sales_loaded", details={"rows": n})
        print(f"[seed] Loaded {n:,} rows from {csv.name}")


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "init"
    if cmd == "init":
        create_schema()
        seed_admin()
        seed_sales()
    elif cmd == "reset":
        if "--yes" not in argv:
            print("Add --yes to confirm. This wipes the database.")
            return 1
        print("[migrate] Dropping all tables")
        Base.metadata.drop_all(bind=engine)
        create_schema()
        seed_admin()
        seed_sales()
    elif cmd == "reseed":
        seed_sales(force=True)
    else:
        print(f"Unknown command: {cmd}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
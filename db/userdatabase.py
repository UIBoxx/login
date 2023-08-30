import databases
import sqlalchemy

DATABASE_URL = "sqlite:///./user.db"
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()


users = sqlalchemy.Table(
    "users",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("username", sqlalchemy.String),
    sqlalchemy.Column("email", sqlalchemy.String),
    sqlalchemy.Column("hashed_password", sqlalchemy.String),
)

workspaces = sqlalchemy.Table(
    "workspaces",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("workspace_name", sqlalchemy.String),
    sqlalchemy.Column("owner_email", sqlalchemy.String),
    sqlalchemy.Column("model_count", sqlalchemy.Integer),
    sqlalchemy.Column("timestamp", sqlalchemy.String),
)

models = sqlalchemy.Table(
    "models",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id")),
    sqlalchemy.Column("model_name", sqlalchemy.String),
    sqlalchemy.Column("model_description", sqlalchemy.String),
    sqlalchemy.Column("model_directory_path", sqlalchemy.String),
)

datasets = sqlalchemy.Table(
    "datasets",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("user_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("users.id")),
    sqlalchemy.Column("dataset_name", sqlalchemy.String),
    sqlalchemy.Column("dataset_description", sqlalchemy.String),
    sqlalchemy.Column("dataset_directory_path", sqlalchemy.String),
)

engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)
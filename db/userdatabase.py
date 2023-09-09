import databases
import sqlalchemy

DATABASE_URL = "sqlite:///./db/user.db"
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

# workspaces = sqlalchemy.Table(
#     "workspaces",
#     metadata,
#     sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
#     sqlalchemy.Column("workspace_name", sqlalchemy.String),
#     sqlalchemy.Column("owner_email", sqlalchemy.String),
#     sqlalchemy.Column("model_count", sqlalchemy.Integer),
#     sqlalchemy.Column("timestamp", sqlalchemy.DateTime(timezone=True), server_default=sqlalchemy.func.now()),
# )


# workspaceData = sqlalchemy.Table(
#     "workspaceData",
#     metadata,
#     sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
#     sqlalchemy.Column("workspace_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("workspaces.id")),
#     sqlalchemy.Column("Type", sqlalchemy.String),
#     sqlalchemy.Column("database_path", sqlalchemy.String),
#     sqlalchemy.Column("model_path", sqlalchemy.String),
#     sqlalchemy.Column("upscaling_path", sqlalchemy.String),
#     sqlalchemy.Column("model_name", sqlalchemy.String),
#     sqlalchemy.Column("is_trained", sqlalchemy.Boolean),
#     sqlalchemy.Column("models_count", sqlalchemy.Integer),
# )

workspaces = sqlalchemy.Table(
    "workspaces",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("workspace_name", sqlalchemy.String),
    sqlalchemy.Column("owner_email", sqlalchemy.String),
    sqlalchemy.Column("model_count", sqlalchemy.Integer),
    sqlalchemy.Column("timestamp", sqlalchemy.String),
)

workspaceData = sqlalchemy.Table(
    "workspaceData",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("workspace_id", sqlalchemy.Integer, sqlalchemy.ForeignKey("workspaces.id")),
    sqlalchemy.Column("Type", sqlalchemy.String),
    sqlalchemy.Column("database_path", sqlalchemy.String),
    sqlalchemy.Column("model_path", sqlalchemy.String),
    sqlalchemy.Column("upscaling_path", sqlalchemy.String),
    sqlalchemy.Column("model_name", sqlalchemy.String),
    sqlalchemy.Column("is_trained", sqlalchemy.Boolean),
    sqlalchemy.Column("models_count", sqlalchemy.Integer),
)


engine = sqlalchemy.create_engine(DATABASE_URL)
metadata.create_all(engine)
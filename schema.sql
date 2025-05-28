CREATE TABLE tbl_event(
    id integer GENERATED ALWAYS AS IDENTITY NOT NULL,
    occurred_at timestamp without time zone,
    name varchar(255),
    attributes jsonb,
    PRIMARY KEY(id)
);
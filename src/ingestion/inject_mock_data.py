from __future__ import annotations

import os
import sqlite3
from contextlib import closing

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Dynamically find the project root and point to the database folder
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
SQLITE_DB_PATH = os.path.join(root_dir, "database", "risk_data.db")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")

def inject_sqlite() -> None:
    """Populate SQLite with mock financial data for a credit risk scenario."""
    with closing(sqlite3.connect(SQLITE_DB_PATH)) as conn:
        cur = conn.cursor()

        # Ensure tables exist (idempotent, matches schemas.py definitions)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS Clients (
                ClientID   INTEGER PRIMARY KEY,
                Name       TEXT,
                Turnover   REAL,
                RiskRating TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS Loans (
                LoanID       INTEGER PRIMARY KEY,
                ClientID     INTEGER,
                Amount       REAL,
                InterestRate REAL,
                Status       TEXT,
                FOREIGN KEY (ClientID) REFERENCES Clients(ClientID)
            );
            """
        )

        # Clear existing data so script is idempotent
        cur.execute("DELETE FROM Loans;")
        cur.execute("DELETE FROM Clients;")

        # Insert mock clients
        clients = [
            (1, "Apex Dynamics", 50_000_000.0, "Low"),
            (2, "Mirage Holdings", 150_000.0, "High"),
            (3, "Vanguard Logistics", 8_500_000.0, "Medium"),
        ]
        cur.executemany(
            "INSERT INTO Clients (ClientID, Name, Turnover, RiskRating) VALUES (?, ?, ?, ?);",
            clients,
        )

        # Insert mock loans
        loans = [
            (101, 1, 2_000_000.0, 5.5, "Performing"),
            (102, 2, 15_000_000.0, 12.0, "Defaulting"),
            (103, 3, 500_000.0, 7.0, "Performing"),
        ]
        cur.executemany(
            "INSERT INTO Loans (LoanID, ClientID, Amount, InterestRate, Status) VALUES (?, ?, ?, ?, ?);",
            loans,
        )

        conn.commit()


def inject_neo4j() -> None:
    """Populate Neo4j with a connected credit risk scenario."""
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")

    if not user or not password:
        raise RuntimeError(
            "NEO4J_USER and NEO4J_PASSWORD environment variables must be set."
        )

    driver = GraphDatabase.driver(NEO4J_URI, auth=(user, password))

    def _inject(tx):
        # Clear the entire graph
        tx.run("MATCH (n) DETACH DELETE n")

        # Companies
        tx.run(
            """
            CREATE (:Company {id: 1, name: 'Apex Dynamics', sector: 'Manufacturing'})
            CREATE (:Company {id: 2, name: 'Mirage Holdings', sector: 'Real Estate'})
            CREATE (:Company {id: 3, name: 'Vanguard Logistics', sector: 'Transport'});
            """
        )

        # Persons
        tx.run(
            """
            CREATE (:Person {id: 1, name: 'Elias Vance', role: 'Ultimate Beneficial Owner'})
            CREATE (:Person {id: 2, name: 'Sarah Chen', role: 'CFO'});
            """
        )

        # Loans as nodes linked to companies
        tx.run(
            """
            MATCH (apex:Company {id: 1}),
                  (mirage:Company {id: 2}),
                  (vanguard:Company {id: 3})
            CREATE (loan101:Loan {id: 101, amount: 2000000.0, status: 'Performing'})
            CREATE (loan102:Loan {id: 102, amount: 15000000.0, status: 'Defaulting'})
            CREATE (loan103:Loan {id: 103, amount: 500000.0, status: 'Performing'})
            CREATE (apex)-[:BORROWS]->(loan101)
            CREATE (mirage)-[:BORROWS]->(loan102)
            CREATE (vanguard)-[:BORROWS]->(loan103);
            """
        )

        # Ownership and management relationships
        tx.run(
            """
            MATCH (apex:Company {id: 1}),
                  (mirage:Company {id: 2}),
                  (vanguard:Company {id: 3}),
                  (elias:Person {id: 1}),
                  (sarah:Person {id: 2})
            CREATE (elias)-[:OWNS {percent: 80}]->(mirage)
            CREATE (elias)-[:BENEFICIAL_OWNER_OF {percent: 25}]->(apex)
            CREATE (sarah)-[:CFO_OF]->(apex)
            CREATE (sarah)-[:BOARD_MEMBER_OF]->(vanguard);
            """
        )

        # Guarantee and supplier relationships
        tx.run(
            """
            MATCH (apex:Company {id: 1}),
                  (mirage:Company {id: 2}),
                  (vanguard:Company {id: 3}),
                  (loan102:Loan {id: 102}),
                  (loan103:Loan {id: 103})
            CREATE (apex)-[:GUARANTEES]->(loan102)
            CREATE (vanguard)-[:GUARANTEES]->(loan103)
            CREATE (apex)-[:SUPPLIER_OF]->(vanguard)
            CREATE (vanguard)-[:LOGISTICS_PROVIDER_FOR]->(mirage);
            """
        )

    try:
        with driver.session() as session:
            session.execute_write(_inject)
    finally:
        driver.close()


def main() -> None:
    load_dotenv(override=True)

    inject_sqlite()
    print(f"SQLite mock data injected into '{SQLITE_DB_PATH}'.")

    inject_neo4j()
    print("Neo4j mock graph data injected.")


if __name__ == "__main__":
    main()


%%writefile app.py

import streamlit as st
import spacy

# Load English word vectors (medium model)
nlp = spacy.load("en_core_web_md")

STOP_WORDS = {"the", "and", "is", "in", "to", "of", "a", "with", "for", "on"}

def tokenize(text):
    try:
        # Tokenization using spaCy
        doc = nlp(text)
        # Extract tokens, remove stop words, and convert to lowercase
        tokens = [token.text.lower() for token in doc if token.text.lower() not in STOP_WORDS and token.is_alpha]
        return tokens
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return []

def preprocess_table_name(table_name):
    # Remove underscores from the table name
    return table_name.replace("_", " ")

def calculate_score_word_embeddings(question, table_name):
    try:
        # Tokenize the question and preprocessed table name
        question_tokens = set(tokenize(question))
        table_tokens = set(tokenize(preprocess_table_name(table_name)))

        # Check if tokens are empty
        if not question_tokens or not table_tokens:
            return 0.0

        # Calculate similarity using word embeddings
        question_vector = nlp(" ".join(question_tokens)).vector
        table_vector = nlp(" ".join(table_tokens)).vector

        # Check if vectors contain NaN values
        if any(map(lambda x: x != x, question_vector)) or any(map(lambda x: x != x, table_vector)):
            return 0.0

        # Calculate cosine similarity
        similarity = question_vector.dot(table_vector) / (question_vector_norm := (question_vector**2).sum())**0.5 / (table_vector_norm := (table_vector**2).sum())**0.5

        return similarity
    except Exception as e:
        print(f"Error during similarity calculation: {e}")
        return 0.0

def search_tables(question, table_names):
    # Calculate relevance scores for each table based on word embeddings similarity
    table_scores = [(table_name, calculate_score_word_embeddings(question, table_name)) for table_name in table_names]

    # Sort tables based on relevance scores in descending order
    sorted_tables = sorted(table_scores, key=lambda x: x[1], reverse=True)

    # Return only the top 5 results
    return sorted_tables[:5]

table_names = [
    "employees", "employee_details", "employee_salaries",
    "departments", "department_employees", "department_budget",
    "customers", "customer_orders", "customer_addresses",
    "products", "product_categories", "product_inventory",
    "suppliers", "supplier_contacts", "supplier_products",
    "orders", "order_items", "order_shipments",
    "transactions", "transaction_details", "transaction_history",
    "invoices", "invoice_items", "invoice_payments",
    "users", "user_profiles", "user_preferences",
    "accounts", "account_transactions", "account_balances",
    "books", "book_authors", "book_genres",
    "authors", "author_books", "author_awards",
    "employees_audit", "employees_backup", "employees_archive",
    "departments_archive", "products_archive", "orders_archive",
    "transactions_archive", "customers_archive", "suppliers_archive",
    "invoices_archive", "users_archive", "accounts_archive",
    "books_archive", "authors_archive",
]

# Streamlit app code
st.title("Table Search Chat")

question = st.text_input("Ask a question:")
if question:
    result = search_tables(question, table_names)
    formatted_result = [f"{i+1}. {table} - Score: {score:.2f}" for i, (table, score) in enumerate(result)]
    st.markdown("\n".join(formatted_result))
import sqlite3
import argparse
import sys
import os

# Try to import readline for a better user experience (command history, editing)
# It's not available on all OS (e.g., standard Windows), so we make it optional.
try:
    import readline
except ImportError:
    print("readline module not found. Command history and editing will be disabled.")
    print("On Windows, you can install it via: pip install pyreadline3")


class InteractiveSQLiteShell:
    """
    An interactive shell for interacting with an SQLite database.
    It mimics the basic functionality of the native sqlite3 command-line tool.
    """

    def __init__(self, db_path: str):
        """
        Initializes the shell and connects to the specified database.

        Args:
            db_path (str): The path to the SQLite database file.
        """
        self.db_path = db_path
        self.connection = None
        self._connect()

    def _connect(self):
        """Establishes the connection to the SQLite database."""
        try:
            # The connection will create the file if it doesn't exist.
            self.connection = sqlite3.connect(self.db_path)
            self.connection.enable_load_extension(True)
            try:
                import sqlite_vec
                import sqlite_lembed
                sqlite_vec.load(self.connection)
                sqlite_lembed.load(self.connection)
                print("sqlite-vec 扩展已成功加载。")
            except (ImportError, sqlite3.Error) as e:
                print(f"警告: 无法加载 sqlite-vec 扩展: {e}")
                print("向量搜索查询可能会失败。")
            # Set row_factory to access columns by name for easier processing
            self.connection.row_factory = sqlite3.Row
            print(f"Successfully connected to database: {self.db_path}")
            print('Enter ".help" for a list of special commands.')
        except sqlite3.Error as e:
            print(f"Error: Could not connect to database '{self.db_path}'.", file=sys.stderr)
            print(f"SQLite error: {e}", file=sys.stderr)
            sys.exit(1)

    def _handle_meta_command(self, command: str) -> bool:
        """
        Handles special (meta) commands that start with a dot.

        Args:
            command (str): The command entered by the user.

        Returns:
            bool: True if the command was a meta-command, False otherwise.
        """
        if command.lower() in (".exit", "quit"):
            self.close()
            print("Exiting the SQLite shell. Goodbye!")
            sys.exit(0)
        elif command.lower() == ".help":
            print("\nSpecial Commands:")
            print("  .tables          - List all tables in the database.")
            print("  .exit or quit    - Exit this interactive shell.")
            print("  .help            - Display this help message.\n")
            return True
        elif command.lower() == ".tables":
            self._execute_query("SELECT name FROM sqlite_master WHERE type='table';")
            return True
        return False

    def _execute_query(self, sql_query: str):
        """
        Executes a given SQL query and prints the results in a formatted table.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql_query)

            results = cursor.fetchall()

            if not results:
                # For commands that don't return rows (INSERT, UPDATE, DELETE),
                # cursor.description will be None.
                if cursor.rowcount > -1:
                    print(f"Query executed successfully. {cursor.rowcount} rows affected.")
                else:
                    print("Query executed successfully. No results to display.")
                return

            # --- Formatting the output ---
            headers = results[0].keys()
            column_widths = {header: len(header) for header in headers}

            # Find the max width for each column
            for row in results:
                for header in headers:
                    cell_width = len(str(row[header]))
                    if cell_width > column_widths[header]:
                        column_widths[header] = cell_width

            # Print headers
            header_line = " | ".join(header.ljust(column_widths[header]) for header in headers)
            print(header_line)
            print("-" * len(header_line))

            # Print rows
            for row in results:
                row_line = " | ".join(str(row[header]).ljust(column_widths[header]) for header in headers)
                print(row_line)

        except sqlite3.Error as e:
            print(f"An error occurred: {e}", file=sys.stderr)

    def run_interactive_loop(self):
        """
        The main loop that reads and processes user input.
        """
        buffer = ""
        prompt = "sqlite> "

        while True:
            try:
                line = input(prompt)
            except EOFError: # Handle Ctrl+D to exit
                print("\nExiting...")
                break

            if not line and not buffer:
                continue

            # Append the new line to the buffer
            buffer += line

            # If the command is a meta-command, handle it immediately
            if buffer.strip().startswith('.'):
                self._handle_meta_command(buffer.strip())
                buffer = ""
                prompt = "sqlite> "
                continue

            # If the buffer ends with a semicolon, it's a complete SQL statement
            if buffer.strip().endswith(';'):
                # Execute the complete SQL command
                self._execute_query(buffer)
                # Reset buffer and prompt for the next command
                buffer = ""
                prompt = "sqlite> "
            else:
                # The command is incomplete, so change the prompt
                prompt = "   ...> "

    def close(self):
        """Closes the database connection if it's open."""
        if self.connection:
            self.connection.close()


def main():
    """
    Main function to parse arguments and start the shell.
    This structure is consistent with the reference file's use of a
    single configuration input to start the main process.
    """
    parser = argparse.ArgumentParser(
        description="An interactive terminal for SQLite databases.",
        epilog="Example: python sqlite_shell.py my_app.db"
    )
    parser.add_argument(
        "db_path",
        type=str,
        help="The path to the SQLite database file to open."
    )
    args = parser.parse_args()

    # Check if the path is a directory
    if os.path.isdir(args.db_path):
        print(f"Error: '{args.db_path}' is a directory. Please provide a path to a file.", file=sys.stderr)
        sys.exit(1)

    shell = InteractiveSQLiteShell(db_path=args.db_path)
    try:
        shell.run_interactive_loop()
    finally:
        shell.close()


if __name__ == "__main__":
    main()
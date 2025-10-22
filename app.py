
import pathway as pw

class MySchema(pw.Schema):

    message: str = pw.column_definition()

data = pw.debug.table_from_rows(
    schema=MySchema,
    rows=[("Hello, World!",)],
)

pw.debug.compute_and_print(data)

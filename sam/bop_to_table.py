import pickle

def display_table(data_dict, col_width=10):
    row_names = sorted(list(set([t for (t, r) in data_dict.keys()])))
    column_names = sorted(list(set([r for (t, r) in data_dict.keys()])))

    # Header row (column names in scientific notation)
    header = " " * col_width  # Empty space for the corner where row and column names meet
    header += "".join(f"{name:.1e}".rjust(col_width) for name in column_names)
    print(header)

    # Print the table
    for row in row_names:
        # Print row header (row name in scientific notation)
        row_str = f"{row:.1e}".rjust(col_width)

        # Print row values
        for col in column_names:
            # (tvt, rvt): metric
            value = data_dict.get((row, col), 0.0)
            row_str += "{:10.2f}".format(value)  # Format the value to 2 decimal places
        # Print the entire row
        print(row_str)

recall_file = '/home/ros/kzorina/vojtas/ycbv/ablation_kz_recall.p'
precision_file = '/home/ros/kzorina/vojtas/ycbv/ablation_kz_precision.p'

recall_data = pickle.load(open(recall_file, 'rb'))
precision_data = pickle.load(open(precision_file, 'rb'))

print("Recall results")
display_table(recall_data)
print("Precision results")
display_table(precision_data)
# Counter variable
n=0

# Iterate over each file in the directory
for file in *; do
    # Check if the file is a regular file
    if [ -f "$file" ]; then
        # Get the file extension
        ext="${file##*.}"
        # Rename the file using the counter variable
        new_name=$(printf "%04d.%s" "$n" "$ext")
        mv "$file" "$new_name"
        # Increment the counter variable
        ((n++))
    fi
done
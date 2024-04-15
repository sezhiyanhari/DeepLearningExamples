# Assign the command line arguments to variables
lower_limit=417824
upper_limit=427576

# Iterate over the range of process IDs and attempt to kill each process
for pid in $(seq $lower_limit $upper_limit); do
  echo "Attempting to kill process $pid"
  kill -9 $pid 2>/dev/null

  if [ $? -eq 0 ]; then
    echo "Process $pid killed successfully."
  else
    echo "Failed to kill process $pid or process does not exist."
  fi
done
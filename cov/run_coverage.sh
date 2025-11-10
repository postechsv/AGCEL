echo "Running test coverage..."
pytest --cov=AGCEL \
       --cov=train \
       --cov=test \
       --cov-report=html \
       --cov-report=term \
       testcases/

echo ""
echo "Coverage report generated in htmlcov/index.html"
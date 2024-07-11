maude='/home/byhoson/maude/maude.linux64'

#python3 train.py benchmarks/onethirdrule/onethirdrule-analysis.maude init3 disagree 1000

#$maude test.maude <<< 'red in TEST-PROP : printResult(search(M, upTerm(init3), disagree)) .' > log

echo 'dfs init7a'
$maude test.maude <<< 'red in TEST-DFS : printResult(search(M, upTerm(init7a), disagree)) .' > log

### print result
grep -IHnr "(# states)" log
echo "# hit:"
grep -o 'hit' log | wc -l
echo "# miss:"
grep -o 'miss' log | wc -l


maude='/home/byhoson/maude/maude.linux64'

python3 train.py benchmarks/onethirdrule/onethirdrule-hs.maude init4 disagree 1000

echo 'qhs init3'
$maude test.maude <<< 'red in TEST-QHS : printResult(search(M, upTerm(init3), disagree)) .' > log
### print result
grep -IHnr "(# states)" log
echo "# hit:"
grep -o 'hit' log | wc -l
echo "# miss:"
grep -o 'miss' log | wc -l

echo 'qhs init4'
$maude test.maude <<< 'red in TEST-QHS : printResult(search(M, upTerm(init4), disagree)) .' > log
### print result
grep -IHnr "(# states)" log
echo "# hit:"
grep -o 'hit' log | wc -l
echo "# miss:"
grep -o 'miss' log | wc -l

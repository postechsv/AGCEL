maude='/home/byhoson/maude/maude.linux64'

#python3 train.py benchmarks/onethirdrule/onethirdrule-analysis.maude init3 disagree 1000

$maude test.maude <<< 'red in TEST-BFS : printResult(search(M, upTerm(init4), disagree)) .' > log

#echo 'qhs init5b'
#$maude test.maude <<< 'red in TEST-QHS : printResult(search(M, upTerm(init6), disagree)) .' > log
### print result
#grep -IHnr "(# states)" log
#echo "# hit:"
#grep -o 'hit' log | wc -l
#echo "# miss:"
#grep -o 'miss' log | wc -l

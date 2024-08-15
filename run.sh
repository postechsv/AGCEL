maude='/home/byhoson/maude/maude.linux64'

#python3 train.py benchmarks/onethirdrule/onethirdrule-analysis.maude init5a 'decideRHS(0)' 1000
#python3 train.py benchmarks/onethirdrule/onethirdrule-analysis.maude init5a 'decideRHS(1)' 1000

#python3 train.py benchmarks/onethirdrule/onethirdrule-analysis.maude init3 disagree 1000

#$maude test.maude <<< 'red in TEST-PROP : printResult(search(M, upTerm(init3), disagree)) .' > log


$maude qhs.maude <<< 'red in TEST-QHS : getNStates(search(M, upTerm(init5a), disagree)) .' > log

### print result
#grep -IHnr "(# states)" log
#echo "# hit:"
#grep -o 'hit' log | wc -l
#echo "# miss:"
#grep -o 'miss' log | wc -l


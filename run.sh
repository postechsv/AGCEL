maude='/home/byhoson/maude/maude.linux64'

#python3 train.py benchmarks/onethirdrule/onethirdrule-hs.maude init3 disagree 50
$maude test.maude <<< 'red in TEST-DFS : getCtx(search(M, upTerm(init3), disagree)) .' > log

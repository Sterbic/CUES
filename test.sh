if [ ! -d "out" ]; then
	mkdir out
fi

./cues data/live_journal_2M_lines_compressed.txt 0 1 1 5 out

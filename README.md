# LDAitkmitl

1. pip install -r requirements.txt

2. You don't have to pip install pylexto, just MOVE "pylexto" and "jpype" folders in your ../anaconda3/lib/python3.7/site-packages/ (before RUN main.py, pls delete these folders)

3. RUN python main.py


<b>If you got an error: marisa-trie/include/marisa/stdio.h:4:10: fatal error: 'cstdio' file not found</b>

* For Mac Mojave

1. Install a command line tool by running the following command in terminal: open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg

https://stackoverflow.com/questions/52509602/cant-compile-c-program-on-a-mac-after-upgrade-to-mojave

2. Add CPATH for gcc environment by running the following command in terminal or adding it to the bash_profile: export CPATH=/Library/Developer/CommandLineTools/usr/include/c++/v1

https://github.com/pytries/marisa-trie/issues/50

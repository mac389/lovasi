import re
import twokenize as Tokenizer

test = "Happy Bday @ImTheFrancescaC You're 10 years old! I wish you have amazing day LYSM pretty #birthday #of #a #princess pic.twitter.com/1s4DiF7TOn"

test = Tokenizer.tokenizeRawTweetText(test)

print test
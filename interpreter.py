#!/usr/bin/python3
"""
interpreter.py
by David Shuckerow (djs0017@auburn.edu)

We can decompose the SPL language into the following:
<Program> -> <Title> <DramatisPersonae> <Acts>
<Title> -> punctuation-terminated string
<DramatisPersonae> -> <Character> | <Character> <DramatisPersonae>
<Acts> -> <Act> | <Act> <Acts>
<Character> -> <Name>, <Description>
<Act> -> Act <Roman>: <Title> <Scenes>
<Scenes> -> <Scene> | <Scene> <Scenes>
<Scene> -> Scene <Roman>: <Title> <Lines>
<Lines> -> <Line> | <Instruction> | <Line> <Lines> | <Instruction> <Lines>
<Line> -> <Name>: <Sentences>
<Sentences> -> <Sentence> | <Sentence> <Sentences>
<Instruction> -> [<Enter>] | [<Exit>] | [<Exeunt>]
<Enter> -> Enter <NameList>
<Exit> -> Exit <Name>
<Exeunt> -> Exeunt | Exeunt <NameList>
<NameList> -> <Name> | <Name> and <NameList>
"""
import sys


PUNCTUATION = ".!?,:;[]"
VERBOSE = False

"""
Abstract Syntax Tree
"""
def tokenize(f):
	contents = open(f).read()
	for p in PUNCTUATION:
		contents = contents.replace(p, " {} ".format(p))
	tokens = contents.split()
	if VERBOSE:
		print("Tokens: "+str(tokens))
	return tokens

def find_punctuation(tokens, punc=PUNCTUATION):
	closest_punctuation = [tokens.index(p) if p in tokens else len(tokens) for p in punc]
	return tokens[:min(closest_punctuation)]


class ASTNode(object):
	
	def __init__(self, tokens):
		self.tokens = tokens

	def parse(self):
		raise Exception("Not Implemented")

	def eval(self, environ):
		raise Exception("Not Implemented")


class Program(ASTNode):
	
	def __init__(self, tokens):
		self.tokens = tokens
		self.title = None
		self.dramatis_personae = []
		self.acts = []

	def parse(self):
		self.title = Title(self.tokens)
		self.title.parse()
		title_end = len(self.title.tokens)+1
		self.dramatis_personae = DramatisPersonae(self.tokens[title_end:])
		self.dramatis_personae.parse()
		dp_end = len(self.dramatis_personae.tokens)
		self.acts = Acts(self.tokens[title_end+dp_end:])
		self.acts.parse()
		if VERBOSE:
			print("Program: {}\n{}\n{}".format(self.title, self.dramatis_personae, self.acts))
	
	def eval(self, environ):
		environ = dict(environ)
		environ["Act"] = 1
		environ["Scene"] = 1
		environ["Goto"] = ("Act", 1)
		environ["Stage"] = set()
		environ = self.title.eval(environ)
		environ = self.dramatis_personae.eval(environ)
		environ = self.acts.eval(environ)
		return environ


class Title(ASTNode):

	def parse(self):
		self.tokens = find_punctuation(self.tokens)
		self.contents = ' '.join(self.tokens)

	def eval(self, environ):
		return environ

	def __str__(self):
		return "Title: {}".format(self.contents)


class DramatisPersonae(ASTNode):

	def parse(self):
		start_index = 0
		self.contents = []
		while self.tokens[start_index] != "Act":
			self.contents.append(Character(self.tokens[start_index:]))
			self.contents[-1].parse()
			start_index += len(self.contents[-1].tokens) + 1
		self.tokens = self.tokens[:start_index]

	def eval(self, environ):
		environ = dict(environ)
		environ["Characters"] = dict(zip([c.name.split()[0] for c in self.contents], 
									     [c.name.split() for c in self.contents]))
		for character in self.contents:
			environ = character.eval(environ)
		return environ

	def __str__(self):
		return "\n".join(map(str, self.contents))


class Character(ASTNode):

	def parse(self):
		self.name = self.tokens[:self.tokens.index(",")]
		self.description = find_punctuation(self.tokens[len(self.name)+1:])
		self.tokens = self.name + [","] + self.description
		self.name = " ".join(self.name)
		self.description = " ".join(self.description)

	def eval(self, environ):
		environ = dict(environ, **{self.name:[0]})
		return environ

	def __str__(self):
		return self.name+", "+self.description+"."


class Acts(ASTNode):

	def parse(self):
		self.contents = []
		start_index = 0
		while start_index < len(self.tokens):
			# create an act and add it to the list of tokens.
			a = Act(self.tokens[start_index:])
			a.parse()
			#if VERBOSE:
				#print(a)
			self.contents.append(a)
			start_index += len(a.tokens) + 1

	def eval(self, environ):
		while environ["Goto"][0] == "Act" and environ["Goto"][1] <= len(self.contents):
			if VERBOSE:
				print(environ["Goto"])
			environ = dict(environ)
			environ["Act"] = environ["Goto"][1]
			environ["Scene"] = 1
			environ["Goto"] = ("None", 0)
			environ = self.contents[environ["Act"]-1].eval(environ)
		return environ

	def __str__(self):
		return "\n".join(map(str, self.contents))


class Act(ASTNode):

	def parse(self):
		self.number = Roman(self.tokens[1])
		self.description = find_punctuation(self.tokens[3:])
		scene_index = len(self.description) + 4
		self.description = " ".join(self.description)
		self.scenes = []
		while scene_index < len(self.tokens) and self.tokens[scene_index] == "Scene":
			s = Scene(self.tokens[scene_index:])
			s.parse()
			scene_index += len(s.tokens)
			self.scenes.append(s)
		self.tokens = self.tokens[:scene_index-1]

	def eval(self, environ):
		environ = dict(environ)
		environ["Scene"] = 1
		#print(environ)
		while environ["Goto"][0] != "Act":
			environ = self.scenes[environ["Scene"]-1].eval(environ)
			if environ["Goto"][0] == "None":
				environ["Scene"] += 1
				if environ["Scene"] > len(self.scenes):
					environ["Goto"] = ("Act", self.number.value + 1)
		return environ

	def __str__(self):
		return "\tAct " + str(self.number) + ": "+self.description+"\n\t"+"\n\t".join(map(str, self.scenes))


class Scene(ASTNode):

	def parse(self):
		self.number = Roman(self.tokens[1])
		#if VERBOSE:
		#	print("Scene: {}".format(self.number))
		self.description = find_punctuation(self.tokens[3:])
		scene_index = len(self.description) + 4
		self.description = " ".join(self.description)
		self.lines = Lines(self.tokens[scene_index:])
		self.lines.parse()
		self.tokens = self.tokens[:scene_index+len(self.lines.tokens)]

	def eval(self, environ):
		if VERBOSE:
			print("Act: {}, Scene: {}".format(environ["Act"], environ["Scene"]))
		environ = self.lines.eval(environ)
		return environ

	def __str__(self):
		return "Scene "+str(self.number)+": "+self.description + "\n" + str(self.lines)


class Lines(ASTNode):

	def parse(self):
		line_index = 0
		self.commands = []
		while line_index < len(self.tokens) and self.tokens[line_index] != "Scene" and self.tokens[line_index] != "Act":
			cmd = None
			if self.tokens[line_index] == "[":
				# we have an Instruction.
				cmd = Instruction(self.tokens[line_index:])
			else:
				# we have a Line.
				cmd = Line(self.tokens[line_index:])
			cmd.parse()
			self.commands.append(cmd)
			line_index += len(cmd.tokens)
		self.tokens = self.tokens[:line_index]

	def eval(self, environ):
		for l in self.commands:
			environ = l.eval(environ)
		return environ

	def __iter__(self):
		for cmd in self.commands:
			yield cmd 

	def __str__(self):
		return "\n".join(map(str, self.commands))


class Instruction(ASTNode):
	types = ["Enter", "Exit", "Exeunt"]
	def parse(self):
		self.instr = self.tokens[1]
		self.characters = []
		name_index = 2
		instr_index = 2
		while self.tokens[instr_index] != "]":
			if self.tokens[instr_index] == "and":
				if instr_index-name_index > 0:
					self.characters.append(" ".join(self.tokens[name_index:instr_index]))
					name_index = instr_index+1
				else:
					# This should be an error.
					pass
			instr_index += 1
		if instr_index-name_index > 0:
			self.characters.append(" ".join(self.tokens[name_index:instr_index]))
		self.tokens = self.tokens[:instr_index+1]

	def eval(self, environ):
		environ = dict(environ)
		if self.instr == "Enter":
			fun = environ["Stage"].add
		else:
			fun = environ["Stage"].remove
		for c in self.characters:
			fun(c)
		if len(self.characters) == 0 and self.instr == "Exeunt":
			environ["Stage"] = set()
		return environ


	def __str__(self):
		return "["+" ".join(self.tokens[1:-1])+"]"


class Line(ASTNode):

	def parse(self):
		#if VERBOSE:
			#print(self.tokens[:2])
		name_length = self.tokens.index(":")
		self.name = " ".join(self.tokens[:name_length])
		self.sentences = Sentences(self.tokens[name_length+1:])
		self.sentences.parse()
		self.tokens = self.tokens[:len(self.sentences.tokens)+name_length+1]

	def eval(self, environ):
		if self.name not in environ["Stage"]:
			# This is an error.
			raise Exception("Runtime Error: Speaking character not on stage: {}\nEnviron: {}".format(self, environ))
			pass
		if len(environ["Stage"]) != 2:
			# This too is an error.
			raise Exception("Runtime Error: Incorrect number of characters on stage: {}\nEnviron: {}".format(self, environ))
		other = None
		for c in environ["Stage"]:
			if c != self.name:
				other = c
		environ = dict(environ, **{"Speaker":self.name, "Listener":other})
		environ = self.sentences.eval(environ)
		return environ

	def __str__(self):
		return "{}: {}".format(self.name, self.sentences)


class Sentences(ASTNode):

	def parse(self):
		self.contents = []
		start = 0
		end = start+len(find_punctuation(self.tokens[start:]))
		while end < len(self.tokens) and self.tokens[end] not in ["]", "[", ":"]:
			tks = self.tokens[start:min(end+1, len(self.tokens))]
			cls = Sentence
			w0 = tks[0].lower()
			if w0 in PERSONAL_NOUNS:
				cls = Assignment
			elif len(tks) > 2 and w0 in inputs and inputs[w0] == inputs[tks[2].lower()]:
				cls = Input
			elif len(tks) > 2 and w0 in outputs and outputs[w0] == outputs[tks[2].lower()]:
				cls = Output
			elif w0 in gotos:
				cls = Goto
			elif w0 in conditions:
				cls = Conditional
			elif w0 in pushes:
				cls = Push
			elif w0 in pops:
				cls = Pop
			elif tks[-1] == "?":
				cls = Query
			if VERBOSE:
				print(cls.__name__)
			sent = cls(tks)
			sent.parse()
			self.contents.append(sent)
			start = end+1
			end = start+len(find_punctuation(self.tokens[start:]))
		self.tokens = self.tokens[:start]

	def eval(self, environ):
		for s in self.contents:
			environ = s.eval(environ)
		return environ

	def __str__(self):
		return " ".join(map(str, self.contents))


class Sentence(ASTNode):

	def parse_math(self, environ, start=0, start_value=0):
		value = 1
		tokens = self.tokens
		while start < len(tokens):
			i = start
			word = tokens[i]
			w = word.lower()
			if w in ADJECTIVES:
				value *= 2
			elif w in NOUNS:
				value *= NOUNS[w]
				#print("Math: "+str(value))
				return value, start + 1
			elif (word in environ["Characters"] and
				  tokens[i:i+len(environ["Characters"][word])] == environ["Characters"][word]):
				i += len(environ["Characters"][word])
				#if VERBOSE:
					#print(word+": "+str(value))
				value *= environ[" ".join(environ["Characters"][word])][-1]
				return value, start + 1
			w2 = w
			if i + 2 < len(tokens):
				w2 = tokens[i+2].lower()
			if w in parameters:
				if w2 not in parameters:
					start += 1
					params = []
					for i in range(parameters[w]):
						val, start = self.parse_math(environ, start, start_value)
						params.append(val)
					#if VERBOSE:
						#print(params)
					return operations[w](*params), start + 1
			start += 1
		return start_value, start

	def parse(self):
		pass

	def eval(self, environ):
		return environ

	def __str__(self):
		return " ".join(self.tokens[:-1]) + self.tokens[-1]

class Assignment(Sentence):

	def eval(self, environ):
		environ = dict(environ)
		listener = environ["Listener"]
		environ[listener][-1] = self.parse_math(environ, 0, environ[listener][-1])[0]
		if VERBOSE:
			print("{}: {}".format(self, environ))
		return environ

class Input(Sentence):

	def eval(self, environ):
		environ = dict(environ)
		return environ

class Output(Sentence):
	
	def eval(self, environ):
		listener = environ["Listener"]
		out = environ[listener][-1]
		if outputs[self.tokens[0].lower()] == 1:
			if VERBOSE:
				print(out)
			if out == 20:
				out = "\n"
			else:
				out = chr(out)
		print(out, end="")
		return environ

class Goto(Sentence):
	pass

class Conditional(Sentence):
	pass

class Push(Sentence):
	pass

class Pop(Sentence):
	pass

class Query(Sentence):
	pass

class Roman(object):

	def __init__(self, num):
		romans = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}
		self.string = num
		self.value = 0
		for n1, n2 in zip(num, num[1:]):
			self.value += -romans[n1] if romans[n1] < romans[n2] else romans[n1]
		self.value += romans[num[-1]]

	def __str__(self):
		return self.string


"""
Keywords
"""
def load_map_from_file(f):
	with open(f) as fi:
		mapping = {}
		for line in fi:
			split = line.split()
			key = " ".join(split[:-1])
			value = int(split[-1])
			mapping[key] = value
		return mapping

outputs = load_map_from_file("keywords/outputs.kws")
inputs = load_map_from_file("keywords/inputs.kws")
parameters = load_map_from_file("keywords/operations.kws")
operations = {
	"sum": lambda x, y: x + y,
	"difference": lambda x, y: x - y,
	"product": lambda x, y: x * y,
	"division": lambda x, y: x // y,
	"quotient": lambda x, y: x // y,
	"square": lambda x: x ** 2,
	"cube": lambda x: x ** 3,
	"root": lambda x: x ** 0.5,
	"twice": lambda x: x * 2,
}
gotos = set([
	"Let"
])
conditions = set([
	"If"
])
pushes = set([
	"Remember"
])
pops = set([
	"Recall"
])
NOUNS = load_map_from_file("keywords/nouns.kws")
ADJECTIVES = load_map_from_file("keywords/adjectives.kws")
PERSONAL_NOUNS = load_map_from_file("keywords/personalnouns.kws")
PERSONAL_ADJECTIVES = load_map_from_file("keywords/personaladjectives.kws")


"""
main
"""
if __name__ == "__main__":
	if "-v" in sys.argv:
		VERBOSE = True
	program = Program(tokenize("../hello.spl"))
	program.parse()
	state = program.eval(dict())
	if VERBOSE:
		print(state)

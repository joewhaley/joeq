PACKAGES = Allocator Assembler Bootstrap ClassLib Clazz Compil3r GC Interpreter Linker Main Run_Time Scheduler Synchronization UTF Util java

ALLDIRS = $(shell find $(PACKAGES) -type d | grep -v CVS)
ALLJAVA = $(foreach d, $(ALLDIRS), $(wildcard $(d)/*.java))
ALLPKGS = $(sort $(foreach j, $(ALLJAVA), $(dir $(j))))
ALLCLASS = $(foreach d, $(ALLDIRS), $(wildcard $(d)/*.class))

CLASSFILE_VERSION = 1.3

all:	javac

### RULES TO BUILD SOURCE CODE
#
jikes:	bootclasspath source_list
	jikes -target $(CLASSFILE_VERSION) -bootclasspath @bootclasspath @.source_list

jikes-pedantic:	bootclasspath source_list
	jikes -target $(CLASSFILE_VERSION) -bootclasspath @bootclasspath +P @.source_list

javac:	source_list _javac
# _javac: Assumes prior existence of the .source_list file.
# Useful on cygwin, because cygwin takes a long time to build the .source_list file.
_javac:	
	javac -target $(CLASSFILE_VERSION) @.source_list


### RULES TO CLEAN
#
clean:
	find . -name "*.class" | xargs rm -f
#	rm -f $(ALLCLASS)

veryclean:
	find . -name "*.class" | xargs rm -f bootclasspath .source_list
#	rm -f bootclasspath .source_list $(ALLCLASS)


### RULES TO ACCESS CVS
#
update:
	( export CVS_RSH=ssh ; cvs update -Pd )

commit:
	( export CVS_RSH=ssh ; cvs commit )


### RULES FOR BUILDING BOOT IMAGES
#
bootstrap:	Main/Bootstrapper.class
	java -Xbootclasspath/a:. -mx380M Main.Bootstrapper

bootstrap-sun14-linux:	javac
	java -ms240M -mx240M Main.Bootstrapper -cl ClassLib/sun14_linux/classlist.txt

bootstrap-sun13-linux:	javac
	java -Xbootclasspath/a:. -ms170M -mx170M Main.Bootstrapper -cl ClassLib/sun13_linux/classlist.txt

bootstrap-ibm13-linux:	javac
	java -Xbootclasspath/a:. -ms230M -mx230M Main.Bootstrapper -cl ClassLib/ibm13_linux/classlist.txt

bootstrap-sun14-win32:	javac
	java -ms180M -mx180M Main.Bootstrapper -cl ClassLib/sun14_win32/classlist.txt

bootstrap-sun13-win32:	javac
	java -Xbootclasspath/a:. -ms120M -mx120M Main.Bootstrapper -cl ClassLib/sun13_win32/classlist.txt


### MISC RULES
#
wc:
	@echo Total, and Top Five files:
	@wc -l $(ALLJAVA) | sort -rn | head -7


### LOCAL RULES (you will never need to make these directly)
#
source_list:
	@echo $(ALLJAVA) | xargs -n 1 echo > .source_list
#	find $(PACKAGES) -name "*.java" > .source_list

Main/Bootstrapper.class:
	$(MAKE) javac

bootclasspath:	Main/GetBootClassPath.class
	java Main.GetBootClassPath > bootclasspath

Main/GetBootClassPath.class:	Main/GetBootClassPath.java
	javac Main/GetBootClassPath.java

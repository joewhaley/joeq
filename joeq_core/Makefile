packages = Allocator Assembler Bootstrap ClassLib Clazz Compil3r Interpreter Linker Main Run_Time Scheduler Synchronization UTF Util java

all:	javac

jikes:	bootclasspath
	find ${packages} -name "*.java" | xargs jikes -target 1.3 -bootclasspath @bootclasspath

jikes-pedantic:	bootclasspath
	find ${packages} -name "*.java" | xargs jikes -target 1.3 -bootclasspath @bootclasspath +P

javac:
	find ${packages} -name "*.java" | xargs javac -target 1.3

clean:
	find ${packages} -name "*.class" | xargs rm -f

veryclean:
	find ${packages} -name "*.class" | xargs rm -f bootclasspath

update:
	( export CVS_RSH=ssh ; cvs update -Pd )

commit:
	( export CVS_RSH=ssh ; cvs commit )

bootstrap:
	java -Xbootclasspath/a:. -mx250M Main.Bootstrapper

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

bootclasspath:	Main/GetBootClassPath.class
	java Main.GetBootClassPath > bootclasspath

Main/GetBootClassPath.class:	Main/GetBootClassPath.java
	javac Main/GetBootClassPath.java

packages = Allocator Assembler Bootstrap ClassLib Clazz Compil3r Interpreter Linker Main Run_Time Scheduler Synchronization UTF Util jq.java

all:	jikes

jikes:	bootclasspath
	find ${packages} -name "*.java" | xargs jikes -bootclasspath @bootclasspath

jikes-pedantic:	bootclasspath
	find ${packages} -name "*.java" | xargs jikes -bootclasspath @bootclasspath -P

javac:
	find ${packages} -name "*.java" | xargs javac

clean:
	find ${packages} -name "*.class" | xargs rm -f jq*.class

veryclean:
	find ${packages} -name "*.class" | xargs rm -f jq*.class GetBootClassPath.class bootclasspath

update:
	( export CVS_RSH=ssh ; cvs update -Pd )

commit:
	( export CVS_RSH=ssh ; cvs commit )

bootstrap:
	java -mx250M Main.Bootstrapper

bootclasspath:	GetBootClassPath.class
	java GetBootClassPath > bootclasspath

GetBootClassPath.class:	GetBootClassPath.java
	javac GetBootClassPath.java
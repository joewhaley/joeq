/*
 * DoTypeAnalysis.java
 *
 * Created on December 20, 2000, 1:18 AM
 *
 */

package Main;

import java.io.PrintStream;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.HashSet;
import java.util.Set;

import Clazz.*;
import Bootstrap.*;
import Run_Time.*;
import Util.*;
import UTF.Utf8;
import Main.jq;

/*
 * @author  John Whaley
 * @version $Id$
 */
public abstract class DoTypeAnalysis {
    
    public static void main(String[] args) throws IOException {
        jq.Bootstrapping = true; // initialize jq
        jq.DontCompile = true;
        jq.boot_types = new java.util.HashSet();

        Unsafe.installRemapper(new Unsafe.Remapper() {
            public int addressOf(Object o) { return 0; }
            public jq_Type getType(Object o) { return Reflection.getJQType(o.getClass()); }
        });
        
        String classpath = System.getProperty("java.class.path")+
                           System.getProperty("path.separator")+
                           System.getProperty("sun.boot.class.path");
        for (Iterator it = PrimordialClassLoader.classpaths(classpath); it.hasNext(); ) {
            String s = (String)it.next();
            PrimordialClassLoader.loader.addToClasspath(s);
        }
        
        Compil3r.BytecodeAnalysis.TypeAnalysis.classesToAnalyze = new HashSet();
        Iterator i = null; String memberName = null;
        jq_Class interfaceCheck = null;
        for (int x=0; x<args.length; ++x) {
            if (args[x].equals("-trace")) {
                Compil3r.BytecodeAnalysis.TypeAnalysis.TypeAnalysisVisitor.ALWAYS_TRACE = true;
                //Compil3r.BytecodeAnalysis.TypeAnalysis.MethodCallSequence.PRINT_MODEL = true;
                Compil3r.BytecodeAnalysis.TypeAnalysis.TRACE_MAIN = true;
                Compil3r.BytecodeAnalysis.TypeAnalysis.TRACE_ITERATION = true;
                Compil3r.BytecodeAnalysis.TypeAnalysis.AnalysisSummary.TRACE_TRIM = true;
            } else
            if (args[x].equals("-dumpsummaries")) {
                Compil3r.BytecodeAnalysis.TypeAnalysis.DUMP_SUMMARY = true;
            } else
            if (args[x].equals("-i")) {
                String interfaceName = args[++x];
                if (interfaceName.endsWith(".class")) interfaceName = interfaceName.substring(0, interfaceName.length()-6);
                String interfaceDesc = "L"+interfaceName+";";
                interfaceCheck = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType(interfaceDesc);
                interfaceCheck.load(); interfaceCheck.verify(); interfaceCheck.prepare();
            } else
            if (args[x].equals("-bypackage")) {
                Compil3r.BytecodeAnalysis.TypeAnalysis.TypeAnalysisVisitor.BY_PACKAGE = true;
            } else
            if (args[x].equals("-maxdepth")) {
                Compil3r.BytecodeAnalysis.TypeAnalysis.TypeAnalysisVisitor.MAX_DEPTH = Integer.parseInt(args[++x]);
            } else
            if (args[x].equals("-full")) {
                Compil3r.BytecodeAnalysis.TypeAnalysis.classesToAnalyze = null;
            } else
            if (args[x].equals("-printmodel")) {
                /*** TEMPORARILY COMMENT OUT UNTIL WE CHECK IN DETOX STUFF. ***
                Compil3r.BytecodeAnalysis.TypeAnalysis.MethodCallSequence.DetoxModelChecker.PRINT_MODEL = true;
                 *** TEMPORARILY COMMENT OUT UNTIL WE CHECK IN DETOX STUFF. ***/
            } else
            if (args[x].equals("-tracemethod")) {
                Compil3r.BytecodeAnalysis.TypeAnalysis.TypeAnalysisVisitor.trace_method_names.add(args[++x]);
            } else
            if (args[x].equals("-buildmodel")) {
                /*** TEMPORARILY COMMENT OUT UNTIL WE CHECK IN DETOX STUFF. ***
                Compil3r.BytecodeAnalysis.TypeAnalysis.MethodCallSequence.modeler =
                    new Compil3r.BytecodeAnalysis.TypeAnalysis.MethodCallSequence.DetoxModelBuilder();
                 *** TEMPORARILY COMMENT OUT UNTIL WE CHECK IN DETOX STUFF. ***/
            } else
            if (args[x].equals("-checkmodel")) {
                /*** TEMPORARILY COMMENT OUT UNTIL WE CHECK IN DETOX STUFF. ***
                Compil3r.BytecodeAnalysis.TypeAnalysis.MethodCallSequence.modeler =
                    new Compil3r.BytecodeAnalysis.TypeAnalysis.MethodCallSequence.DetoxModelChecker();
                 *** TEMPORARILY COMMENT OUT UNTIL WE CHECK IN DETOX STUFF. ***/
            } else
            if (args[x].equals("-file")) {
                BufferedReader br = new BufferedReader(new FileReader(args[++x]));
                LinkedList list = new LinkedList();
                for (;;) {
                    String s = br.readLine();
                    if (s == null) break;
                    if (s.length() == 0) continue;
                    if (s.startsWith("%")) continue;
                    list.add(s);
                }
                i = new AppendIterator(list.iterator(), i);
            } else
            if (args[x].endsWith("*")) {
                i = new AppendIterator(PrimordialClassLoader.loader.listPackage(args[x].substring(0, args[x].length()-1)), i);
            } else {
                int j = args[x].indexOf('.');
                String classname;
                if ((j != -1) && !args[x].endsWith(".class")) {
                    classname = args[x].substring(0, j);
                    memberName = args[x].substring(j+1);
                } else {
                    classname = args[x];
                }
                i = new AppendIterator(new SingletonIterator(classname), i);
            }
        }
        if (i == null) i = new SingletonIterator("jq.class");
        
        LinkedList classes = new LinkedList();
        while (i.hasNext()) {
            String classname = (String)i.next();
            if (classname.endsWith(".properties")) continue;
            if (classname.endsWith(".class")) classname = classname.substring(0, classname.length()-6);
            String classdesc = "L"+classname+";";
            jq_Class c = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType(classdesc);
            c.load(); c.verify(); c.prepare();
            classes.add(c);
        }
        
        Set classes2 = Compil3r.BytecodeAnalysis.TypeAnalysis.classesToAnalyze;
        if (classes2 != null) {
            classes2.addAll(classes);
        }
        System.out.println("Analyzing these "+classes.size()+" classes: "+classes);
        
        i = classes.iterator();
        while (i.hasNext()) {
            jq_Class c = (jq_Class)i.next();
            if (interfaceCheck != null) {
                if (!TypeCheck.isAssignable(c, interfaceCheck)) {
                    System.out.println(c+" does not implement "+interfaceCheck+", skipping.");
                    continue;
                }
            }
            doClass(System.out, c, memberName);
        }
        try {
            Compil3r.BytecodeAnalysis.TypeAnalysis.dump();
        } catch (java.io.IOException x) { x.printStackTrace(); }
    }

    public static void doClass(PrintStream out, jq_Class c, String memberName) {
        System.out.println("Loading "+c+"...");
        c.load();
        System.out.println("Verifying "+c+"...");
        c.verify();
        System.out.println("Preparing "+c+"...");
        c.prepare();
        System.out.println("Initializing static fields of "+c+"...");
        c.sf_initialize();
        Iterator it;
        for(it = new ArrayIterator(c.getDeclaredStaticMethods());
            it.hasNext(); ) {
            jq_StaticMethod m = (jq_StaticMethod)it.next();
            if (m.getBytecode() == null) continue;
            if (memberName == null || m.getName().toString().equals(memberName))
            {
                out.println(m.toString());
                Compil3r.BytecodeAnalysis.TypeAnalysis.analyze(m);
            }
        }
        for(it = new ArrayIterator(c.getDeclaredInstanceMethods());
            it.hasNext(); ) {
            jq_InstanceMethod m = (jq_InstanceMethod)it.next();
            if (m.isAbstract()) continue;
            if (m.getBytecode() == null) continue;
            if (memberName == null || m.getName().toString().equals(memberName))
            {
                out.println(m.toString());
                Compil3r.BytecodeAnalysis.TypeAnalysis.analyze(m);
            }
        }
    }
    
}

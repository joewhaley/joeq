/*
 * Driver.java
 *
 * Created on January 9, 2002, 9:17 AM
 *
 * @author  Administrator
 * @version 
 */

package Main;

import Clazz.*;
import Compil3r.Quad.*;
import java.io.*;
import java.util.*;
import jq;
import Run_Time.*;
import Bootstrap.*;

public abstract class Driver {

    public static void main(String[] args) {
        // initialize jq
        jq.initializeForHostJVMExecution();
        
        if ((args.length == 0) || args[0].equals("-i")) {
            // interactive mode
            BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
            for (;;) {
                String[] commands;
                try {
                    System.out.print("joeq> ");
                    String line = in.readLine();
                    if (line == null) return;
                    StringTokenizer st = new StringTokenizer(line);
                    int size = st.countTokens();
                    commands = new String[size];
                    for (int i=0; i<size; ++i) {
                        commands[i] = st.nextToken();
                    }
                    jq.assert(!st.hasMoreTokens());
                } catch (IOException x) {
                    System.err.println(x.toString());
                    return;
                }
                for (int i=0; i<commands.length; ++i) {
                    i = processCommand(commands, i);
                }
            }
        }
        for (int i=0; i<args.length; ++i) {
            i = processCommand(args, i);
        }
    }

    static List classesToProcess = new LinkedList();
    static boolean trace_bb = false;
    static boolean trace_method = false;
    static boolean trace_type = false;
    
    public static int processCommand(String[] commandBuffer, int index) {
        try {
            if (commandBuffer[index].equalsIgnoreCase("addtoclasspath")) {
                String path = commandBuffer[++index];
                PrimordialClassLoader.loader.addToClasspath(path);
            } else if (commandBuffer[index].equalsIgnoreCase("trace")) {
                String which = commandBuffer[++index];
                if (which.equalsIgnoreCase("bb")) {
                    trace_bb = true;
                } else if (which.equalsIgnoreCase("method")) {
                    trace_method = true;
                } else if (which.equalsIgnoreCase("type")) {
                    trace_type = true;
                } else {
                    System.err.println("Unknown trace option "+which);
                }
            } else if (commandBuffer[index].equalsIgnoreCase("class")) {
                String canonicalClassName = canonicalizeClassName(commandBuffer[++index]);
                try {
                    jq_Class c = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType(canonicalClassName);
                    c.load();
                    classesToProcess.add(c);
                } catch (NoClassDefFoundError x) {
                    System.err.println("Class "+commandBuffer[index]+" (canonical name "+canonicalClassName+") not found.");
                }
            } else if (commandBuffer[index].equalsIgnoreCase("runpass")) {
                String passname = commandBuffer[++index];
                jq_TypeVisitor cv = null; jq_MethodVisitor mv = null; BasicBlockVisitor bbv = null; QuadVisitor qv = null;
                Object o;
                try {
                    Class c = Class.forName(passname);
                    o = c.newInstance();
                    if (o instanceof jq_TypeVisitor) {
                        cv = (jq_TypeVisitor)o;
                    } else {
                        if (o instanceof jq_MethodVisitor) {
                            mv = (jq_MethodVisitor)o;
                        } else {
                            if (o instanceof BasicBlockVisitor) {
                                bbv = (BasicBlockVisitor)o;
                            } else {
                                if (o instanceof QuadVisitor) {
                                    qv = (QuadVisitor)o;
                                } else {
                                    System.err.println("Unknown pass type "+c);
                                    return index;
                                }
                                bbv = new QuadVisitor.AllQuadVisitor(qv, trace_bb);
                            }
                            mv = new BasicBlockVisitor.AllBasicBlockVisitor(bbv, trace_method);
                        }
                        cv = new jq_MethodVisitor.DeclaredMethodVisitor(mv, trace_type);
                    }
                } catch (java.lang.ClassNotFoundException x) {
                    System.err.println("Cannot find pass named "+passname+".");
                    System.err.println("Check your classpath and make sure you compiled your pass.");
                    return index;
                } catch (java.lang.InstantiationException x) {
                    System.err.println("Cannot instantiate pass "+passname+": "+x);
                    return index;
                } catch (java.lang.IllegalAccessException x) {
                    System.err.println("Cannot access pass "+passname+": "+x);
                    System.err.println("Be sure that you made your class public?");
                    return index;
                }
                for (Iterator i = classesToProcess.iterator(); i.hasNext(); ) {
                    jq_Type t = (jq_Type)i.next();
                    t.accept(cv);
                }
                System.out.println("Completed pass! "+o);
            } else if (commandBuffer[index].equalsIgnoreCase("exit") || commandBuffer[index].equalsIgnoreCase("quit")) {
                System.exit(0);
            } else {
		int index2 = TraceFlags.setTraceFlag(commandBuffer, index);
		if (index == index2)
		    System.err.println("Unknown command "+commandBuffer[index++]);
            }
        } catch (ArrayIndexOutOfBoundsException x) {
            System.err.println("Incomplete command");
        }
        return index;
    }
    public static String canonicalizeClassName(String s) {
        if (s.endsWith(".class")) s = s.substring(0, s.length()-6);
        s = s.replace('.', '/');
        String desc = "L"+s+";";
        return desc;
    }
    
}

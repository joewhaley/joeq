// JoeqVM.java, created Sat Dec 14  2:52:34 2002 by mcmartin
// Copyright (C) 2001-3 jwhaley
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package Main;

import java.util.Iterator;

import Allocator.SimpleAllocator;
import Bootstrap.MethodInvocation;
import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import Clazz.jq_Class;
import Clazz.jq_NameAndDesc;
import Clazz.jq_StaticMethod;
import Compil3r.CompilationState;
import Compil3r.CompilationState.DynamicCompilation;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Scheduler.jq_MainThread;
import Scheduler.jq_NativeThread;
import Scheduler.jq_Thread;
import UTF.Utf8;
import Util.Assert;
import Util.Strings;

/**
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class JoeqVM {
    public static void boot() throws Throwable {
        try {
            // initialize the thread data structures, allocators, etc.
            jq_NativeThread.initInitialNativeThread();

            // init the ctrl-break handler thread.
            jq_NativeThread.initBreakThread();

            // init the garbage collector thread & set it as daemon
            jq_NativeThread.initGCThread();

            // initialize dynamic compiler
            CompilationState.DEFAULT = new DynamicCompilation();
            
            // call java.lang.System.initializeSystemClass()
            ClassLibInterface.DEFAULT.initializeSystemClass();

        } catch (Throwable x) {
            SystemInterface.debugwriteln("Exception occurred during virtual machine initialization");
            SystemInterface.debugwriteln("Exception: " + x);
            if (System.err != null) x.printStackTrace(System.err);
            return;
        }
        int numOfArgs = SystemInterface.main_argc();
        String[] args = new String[numOfArgs];
        for (int i = 0; i < numOfArgs; ++i) {
            int len = SystemInterface.main_argv_length(i);
            byte[] b = new byte[len];
            SystemInterface.main_argv(i, b);
            args[i] = new String(b);
        }
        String classpath = ".";
        int i = 0;
        for (; ;) {
            if (i == args.length) {
                printUsage();
                return;
            }
            if (args[i].equals("-cp") || args[i].equals("-classpath")) { // class path
                classpath = args[++i];
                ++i;
                // update classpath here.
                if (classpath != null) {
                    Iterator it = PrimordialClassLoader.classpaths(classpath);
                    while (it.hasNext()) {
                        String s = (String) it.next();
                        PrimordialClassLoader.loader.addToClasspath(s);
                    }
                }
                continue;
            }
            if (args[i].equals("-nt") || args[i].equals("-native_threads")) { // number of native threads
                jq.NumOfNativeThreads = Integer.parseInt(args[++i]);
                ++i;
                continue;
            }
            if (args[i].startsWith("-mx")) { // max memory
                String amt = args[i].substring(3);
                int mult = 1;
                if (amt.endsWith("m") || amt.endsWith("M")) {
                    mult = 1048576;
                    amt = amt.substring(0, amt.length()-1);
                } else if (amt.endsWith("k") || amt.endsWith("K")) {
                    mult = 1024;
                    amt = amt.substring(0, amt.length()-1);
                }
                int size = mult * Integer.parseInt(amt);
                //size = HeapAddress.align(size, 20);
                SimpleAllocator.MAX_MEMORY = size;
                ++i;
                continue;
            }
            // todo: other command line switches to change VM behavior.
            int j = TraceFlags.setTraceFlag(args, i);
            if (i != j) {
                i = j;
                continue;
            }
            break;
        }
        if (jq.on_vm_startup != null) {
            Iterator it = jq.on_vm_startup.iterator();
            while (it.hasNext()) {
                MethodInvocation mi = (MethodInvocation) it.next();
                try {
                    mi.invoke();
                } catch (Throwable x) {
                    SystemInterface.debugwriteln("Exception occurred while initializing the virtual machine");
                    SystemInterface.debugwriteln(x.toString());
                    x.printStackTrace(System.err);
                    //return;
                }
            }
        }

        jq_Thread tb = Unsafe.getThreadBlock();
        jq_NativeThread nt = tb.getNativeThread();
        jq_NativeThread.initNativeThreads(nt, jq.NumOfNativeThreads);

        // Here we start method replacement of classes whose name were given as arguments to -replace on the cmd line.
        if (Clazz.jq_Class.TRACE_REPLACE_CLASS) SystemInterface.debugwriteln(Strings.lineSep+"STARTING REPLACEMENT of classes: " + Clazz.jq_Class.classToReplace);
        for (Iterator it = Clazz.jq_Class.classToReplace.iterator(); it.hasNext();) {
            String newCName = (String) it.next();
            PrimordialClassLoader.loader.replaceClass(newCName);
        }
        if (Clazz.jq_Class.TRACE_REPLACE_CLASS) SystemInterface.debugwriteln(Strings.lineSep+"DONE with Classes Replacement!");

        String className = args[i];
        jq_Class main_class = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("L" + className.replace('.', '/') + ";");
        main_class.load();
        jq_StaticMethod main_method = main_class.getStaticMethod(new jq_NameAndDesc(Utf8.get("main"), Utf8.get("([Ljava/lang/String;)V")));
        if (main_method == null) {
            System.err.println("Class " + className + " does not contain a main method!");
            return;
        }
        if (!main_method.isPublic()) {
            System.err.println("Method " + main_method + " is not public!");
            return;
        }
        main_class.cls_initialize();
        String[] main_args = new String[args.length - i - 1];
        System.arraycopy(args, i + 1, main_args, 0, main_args.length);

        //jq_CompiledCode main_cc = main_method.getDefaultCompiledVersion();
        //Reflection.invokestatic_V(main_method, main_args);
        jq_MainThread mt = new jq_MainThread(main_method, main_args);
        mt.start();
        jq_NativeThread.startNativeThreads();
        nt.nativeThreadEntry();
        Assert.UNREACHABLE();
    }

    public static void printUsage() {
        System.out.println("Usage: joeq <classname> <parameters>");
    }
}

// Bootstrapper.java, created Mon Feb  5 23:23:21 2001 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package joeq.Main;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Set;

import joeq.Allocator.CodeAllocator;
import joeq.Allocator.DefaultCodeAllocator;
import joeq.Bootstrap.BootImage;
import joeq.Bootstrap.BootstrapCodeAddress;
import joeq.Bootstrap.BootstrapCodeAllocator;
import joeq.Bootstrap.BootstrapRootSet;
import joeq.Runtime.ObjectTraverser;
import joeq.Class.PrimordialClassLoader;
import joeq.Bootstrap.BootstrapCodeAddress.BootstrapCodeAddressFactory;
import joeq.ClassLib.ClassLibInterface;
import joeq.Class.Delegates;
import joeq.Class.jq_Array;
import joeq.Class.jq_Class;
import joeq.Class.jq_Member;
import joeq.Class.jq_Method;
import joeq.Class.jq_Reference;
import joeq.Class.jq_StaticField;
import joeq.Class.jq_StaticMethod;
import joeq.Class.jq_Type;
import joeq.Compiler.CompilationState;
import joeq.Compiler.BytecodeAnalysis.Trimmer;
import joeq.Compiler.CompilationState.BootstrapCompilation;
import joeq.Memory.CodeAddress;
import joeq.Memory.HeapAddress;
import joeq.Runtime.Reflection;
import joeq.Runtime.SystemInterface;
import joeq.Runtime.Unsafe;
import joeq.UTF.Utf8;
import joeq.Util.Assert;
import joeq.Util.Collections.LinearSet;
import joeq.Util.IO.DirectBufferedFileOutputStream;
import joeq.Util.IO.ExtendedDataOutput;

/*
 * @author  John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public abstract class Bootstrapper {

    private static BootImage objmap;
    
    public static void main(String[] args) throws IOException {
        
        if (jq.RunningNative) {
            System.err.println("Error: self-bootstrapping not supported (yet)");
            System.exit(-1);
        }
        
        String imageName = "jq.obj";
        //int startAddress = 0x00890000;
        String rootMethodClassName = "joeq.Main.JoeqVM";
        String rootMethodName = "boot";
        String classList = null;
        String addToClassList = null;
        boolean TrimAllTypes = false;
        boolean DUMP_COFF = false;
        boolean USE_BYTECODE_TRIMMER = true;

        // initialize list of methods to invoke on joeq startup
        jq.on_vm_startup = new LinkedList();
        
        CodeAddress.FACTORY = joeq.Bootstrap.BootstrapCodeAddress.FACTORY;
        HeapAddress.FACTORY = joeq.Bootstrap.BootstrapHeapAddress.FACTORY;
        //StackAddress.FACTORY = joeq.Bootstrap.BootstrapStackAddress.FACTORY;
        
        jq.IsBootstrapping = true;
        ClassLibInterface.useJoeqClasslib(true);
        
        CodeAllocator.initializeCompiledMethodMap();
        
        if (ClassLibInterface.DEFAULT.getClass().toString().indexOf("win32") != -1) {
            DUMP_COFF = true;
        } else {
            DUMP_COFF = false;
        }
        String osarch = System.getProperty("os.arch");
        if (osarch.equals("i386") || osarch.equals("x86")) {
            try {
                Class.forName("joeq.Scheduler.jq_x86RegisterState");
            } catch (ClassNotFoundException e) {
                System.err.println("Error: cannot load x86 module");
                System.exit(-1);
            }
            String default_compiler_name = System.getProperty("joeq.compiler", "joeq.Compiler.Reference.x86.x86ReferenceCompiler$Factory");
            Delegates.setDefaultCompiler(default_compiler_name);
        } else {
            System.err.println("Error: architecture "+osarch+" is not yet supported.");
            System.exit(-1);
        }

        String classpath = System.getProperty("sun.boot.class.path")+
                           System.getProperty("path.separator")+
                           System.getProperty("java.class.path");
        
        for (int i=0; i<args.length; ) {
            int j = TraceFlags.setTraceFlag(args, i);
            if (i != j) { i = j; continue; }
            if (args[i].equals("-o")) { // output file
                imageName = args[++i];
                ++i; continue;
            }
            if (args[i].equals("-r")) { // root method
                String s = args[++i];
                int dotloc = s.lastIndexOf('.');
                rootMethodName = s.substring(dotloc+1);
                rootMethodClassName = s.substring(0, dotloc);
                ++i; continue;
            }
            if (args[i].equals("-cp") || args[i].equals("-classpath")) { // class path
                classpath = args[++i];
                ++i; continue;
            }
            if (args[i].equals("-cl") || args[i].equals("-classlist")) { // class path
                classList = args[++i];
                ++i; continue;
            }
            if (args[i].equals("-a2cl") || args[i].equals("-addtoclasslist")) { // class path
                addToClassList = args[++i];
                ++i; continue;
            }
            if (args[i].equals("-t")) { // trim all types
                TrimAllTypes = true;
                ++i; continue;
            }
            if (args[i].equalsIgnoreCase("-borland")) {
                BootImage.USE_MICROSOFT_STYLE_MUNGE = false;
                ++i; continue;
            }
            if (args[i].equalsIgnoreCase("-microsoft")) {
                BootImage.USE_MICROSOFT_STYLE_MUNGE = true;
                ++i; continue;
            }
            /*
            if (args[i].equals("-s")) { // start address
                startAddress = Integer.parseInt(args[++i], 16);
                ++i; continue;
            }
             */
            err("unknown command line argument: "+args[i]);
        }
        
        rootMethodClassName = rootMethodClassName.replace('.','/');

        System.out.println("Bootstrapping into "+imageName+", "+(DUMP_COFF?"COFF":"ELF")+" format, root method "+rootMethodClassName+"."+rootMethodName+(TrimAllTypes?", trimming all types.":"."));
        
        for (Iterator it = PrimordialClassLoader.classpaths(classpath); it.hasNext(); ) {
            String s = (String)it.next();
            PrimordialClassLoader.loader.addToClasspath(s);
        }
        
        //Set nullStaticFields = ClassLibInterface.i.bootstrapNullStaticFields();
        //Set nullInstanceFields = ClassLibInterface.i.bootstrapNullInstanceFields();
        //System.out.println("Null static fields: "+nullStaticFields);
        //System.out.println("Null instance fields: "+nullInstanceFields);

        // install bootstrap code allocator
        BootstrapCodeAllocator bca = BootstrapCodeAllocator.DEFAULT;
        DefaultCodeAllocator.default_allocator = bca;
        CodeAddress.FACTORY = BootstrapCodeAddress.FACTORY = new BootstrapCodeAddressFactory(bca);
        bca.init();
        
        // install object mapper
        //ObjectTraverser obj_trav = new ObjectTraverser(nullStaticFields, nullInstanceFields);
        ObjectTraverser obj_trav = ClassLibInterface.DEFAULT.getObjectTraverser();
        Reflection.obj_trav = obj_trav;
        obj_trav.initialize();
        //objmap = new BootImage(bca);
        objmap = BootImage.DEFAULT;
        //HeapAddress.FACTORY = BootstrapHeapAddress.FACTORY = new BootstrapHeapAddressFactory(objmap);
        
        long starttime = System.currentTimeMillis();
        jq_Class c;
        c = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("L"+rootMethodClassName+";");
        c.prepare();
        long loadtime = System.currentTimeMillis() - starttime;

        jq_StaticMethod rootm = null;
        Utf8 rootm_name = Utf8.get(rootMethodName);
        for(Iterator it = Arrays.asList(c.getDeclaredStaticMethods()).iterator();
            it.hasNext(); ) {
            jq_StaticMethod m = (jq_StaticMethod)it.next();
            if (m.getName() == rootm_name) {
                rootm = m;
                break;
            }
        }
        if (rootm == null)
            err("root method not found: "+rootMethodClassName+"."+rootMethodName);
        
        Set classset = new HashSet();
        Set methodset;
        
        starttime = System.currentTimeMillis();
        if (addToClassList != null) {
            BufferedReader dis = new BufferedReader(new FileReader(addToClassList));
            for (;;) {
                String classname = dis.readLine();
                if (classname == null) break;
                if (classname.charAt(0) == '#') continue;
                jq_Type t = PrimordialClassLoader.loader.getOrCreateBSType(classname);
                t.prepare();
                classset.add(t);
            }
        }
        if (classList != null) {
            BufferedReader dis = new BufferedReader(new FileReader(classList));
            for (;;) {
                String classname = dis.readLine();
                if (classname == null) break;
                if (classname.equals("")) continue;
                if (classname.charAt(0) == '#') continue;
                if (classname.endsWith("*")) {
                    Assert._assert(classname.startsWith("L"));
                    Iterator i = PrimordialClassLoader.loader.listPackage(classname.substring(1, classname.length()-1));
                    while (i.hasNext()) {
                        String s = (String)i.next();
                        Assert._assert(s.endsWith(".class"));
                        s = "L"+s.substring(0, s.length()-6)+";";
                        jq_Class t = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType(s);
                        t.prepare();
                        classset.add(t);
                        for (;;) {
                            jq_Array q = t.getArrayTypeForElementType();
                            q.prepare();
                            classset.add(q);
                            t = t.getSuperclass();
                            if (t == null) break;
                            classset.add(t);
                        }
                    }
                } else {
                    jq_Type t = PrimordialClassLoader.loader.getOrCreateBSType(classname);
                    t.prepare();
                    classset.add(t);
                    if (t instanceof jq_Class) {
                        jq_Class q = (jq_Class) t;
                        for (;;) {
                            t = q.getArrayTypeForElementType();
                            t.prepare();
                            classset.add(t);
                            q = q.getSuperclass();
                            if (q == null) break;
                            classset.add(q);
                        }
                    }
                }
            }
            methodset = new HashSet();
            Iterator i = classset.iterator();
            while (i.hasNext()) {
                jq_Type t = (jq_Type)i.next();
                if (t.isClassType()) {
                    jq_Class cl = (jq_Class)t;
                    jq_Method[] ms = cl.getDeclaredStaticMethods();
                    for (int k=0; k<ms.length; ++k) {
                        methodset.add(ms[k]);
                    }
                    ms = cl.getDeclaredInstanceMethods();
                    for (int k=0; k<ms.length; ++k) {
                        methodset.add(ms[k]);
                    }
                    ms = cl.getVirtualMethods();
                    for (int k=0; k<ms.length; ++k) {
                        methodset.add(ms[k]);
                    }
                }
            }
        } else {
            // traverse the code and data starting at the root set to find all necessary
            // classes and members.
            
            if (USE_BYTECODE_TRIMMER) {
                Trimmer trim = new Trimmer(rootm, classset, !TrimAllTypes);
                trim.go();

                BootstrapRootSet rs = trim.getRootSet();
                System.out.println("Number of instantiated types: "+rs.getInstantiatedTypes().size());
                //System.out.println("Instantiated types: "+rs.getInstantiatedTypes());

                System.out.println("Number of necessary methods: "+rs.getNecessaryMethods().size());
                //System.out.println("Necessary methods: "+rs.getNecessaryMethods());

                System.out.println("Number of necessary fields: "+rs.getNecessaryFields().size());
                //System.out.println("Necessary fields: "+rs.getNecessaryFields());
                
                // find all used classes.
                classset = rs.getNecessaryTypes();

                System.out.println("Number of necessary classes: "+classset.size());
                //System.out.println("Necessary classes: "+classset);

                if (TrimAllTypes) {
                    // Trim all the types.
                    Iterator it = classset.iterator();
                    while (it.hasNext()) {
                        jq_Type t = (jq_Type)it.next();
                        System.out.println("Trimming type: "+t.getName());
                        Assert._assert(t.isPrepared());
                        if (t.isClassType()) {
                            rs.trimClass((jq_Class)t);
                        }
                    }
                    System.out.println("Number of instance fields kept: "+jq_Class.NumOfIFieldsKept);
                    System.out.println("Number of static fields kept: "+jq_Class.NumOfSFieldsKept);
                    System.out.println("Number of instance methods kept: "+jq_Class.NumOfIMethodsKept);
                    System.out.println("Number of static methods kept: "+jq_Class.NumOfSMethodsKept);

                    System.out.println("Number of instance fields eliminated: "+jq_Class.NumOfIFieldsEliminated);
                    System.out.println("Number of static fields eliminated: "+jq_Class.NumOfSFieldsEliminated);
                    System.out.println("Number of instance methods eliminated: "+jq_Class.NumOfIMethodsEliminated);
                    System.out.println("Number of static methods eliminated: "+jq_Class.NumOfSMethodsEliminated);
                }

                methodset = rs.getNecessaryMethods();
            } else {
                // TODO: use a supplied call graph.
                BootstrapRootSet rs = null;
                methodset = rs.getNecessaryMethods();
            }
        }
        loadtime += System.currentTimeMillis() - starttime;
        System.out.println("Load time: "+loadtime/1000f+"s");
        
        if (classList == null) {
            dumpClassSet(classset);
            dumpMethodSet(methodset);
        }
            
        // initialize the set of boot types
        objmap.boot_types = classset;
        BootstrapCompilation comp = (BootstrapCompilation) CompilationState.DEFAULT;
        comp.setBootTypes(classset);
        
        if (false) {
            ArrayList class_list = new ArrayList(classset);
            Collections.sort(class_list, new Comparator() {
                public int compare(Object o1, Object o2) {
                    return ((jq_Type)o1).getDesc().toString().compareTo(((jq_Type)o2).getDesc().toString());
                }
                public boolean equals(Object o1, Object o2) { return o1 == o2; }
            });
            System.out.println("Types:");
            Set packages = new LinearSet();
            Iterator it = class_list.iterator();
            while (it.hasNext()) {
                jq_Type t = (jq_Type)it.next();
                String s = t.getDesc().toString();
                System.out.println(s);
                if (s.charAt(0) == 'L') {
                    int index = s.lastIndexOf('/');
                    if (index == -1) s = "";
                    else s = s.substring(1, index+1);
                    packages.add(s);
                }
            }
            System.out.println("Packages:");
            it = packages.iterator();
            while (it.hasNext()) {
                System.out.println("L"+it.next()+"*");
            }
        }
        
        // enable allocations
        objmap.enableAllocations();

        // allocate entrypoints first in bootimage.
        // NOTE: will only be first if java.lang.Object doesn't have any static members.
        SystemInterface._class.sf_initialize();

        //jq.Assert(SystemInterface._entry.getAddress() == startAddress + ARRAY_HEADER_SIZE,
        //          "entrypoint is at "+Strings.hex8(SystemInterface._entry.getAddress()));
        
        // initialize the static fields for all the necessary types
        starttime = System.currentTimeMillis();
        Iterator it = classset.iterator();
        while (it.hasNext()) {
            jq_Type t = (jq_Type)it.next();
            Assert._assert(t.isPrepared());
            t.sf_initialize();
            // initialize static field values, too.
            if (t.isClassType()) {
                jq_Class k = (jq_Class)t;
                Assert._assert((k.getSuperclass() == null) || classset.contains(k.getSuperclass()),
                          k.getSuperclass()+" (superclass of "+k+") is not in class set!");
                jq_StaticField[] sfs = k.getDeclaredStaticFields();
                for (int j=0; j<sfs.length; ++j) {
                    jq_StaticField sf = sfs[j];
                    //System.out.println("Initializing static field: "+sf);
                    objmap.initStaticField(sf);
                    objmap.addStaticFieldReloc(sf);
                }
            }
        }
        long sfinittime = System.currentTimeMillis() - starttime;
        System.out.println("SF init time: "+sfinittime/1000f+"s");
        
        // turn on jq.RunningNative flag in image
        jq_Class jq_class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljoeq/Main/jq;");
        jq_class.setStaticData(jq_class.getOrCreateStaticField("RunningNative","Z"), 1);
        // turn off jq.IsBootstrapping flag in image
        jq_class.setStaticData(jq_class.getOrCreateStaticField("IsBootstrapping","Z"), 0);

        // compile versions of all necessary methods.
        starttime = System.currentTimeMillis();
        it = methodset.iterator();
        while (it.hasNext()) {
            jq_Member m = (jq_Member)it.next();
            if (m instanceof jq_Method) {
                jq_Method m2 = ((jq_Method)m);
                if (m2.getDeclaringClass() == Unsafe._class) continue;
                if (m2.getDeclaringClass().isAddressType()) continue;
                m2.compile();
            }
        }
        long compiletime = System.currentTimeMillis() - starttime;
        System.out.println("Compile time: "+compiletime/1000f+"s");

        // initialize and add the jq_Class/jq_Array/jq_Primitive objects for all
        // necessary types.
        starttime = System.currentTimeMillis();
        it = classset.iterator();
        while (it.hasNext()) {
            jq_Type t = (jq_Type)it.next();
            Assert._assert(t.isSFInitialized());
            
            if (t == Unsafe._class) continue;
            //System.out.println("Compiling type: "+t.getName());
            t.compile();
            t.cls_initialize();
            objmap.getOrAllocateObject(t);
        }
        
        // get the JDK type of each of the classes that could be in our image, so
        // that we can trigger each of their <clinit> methods, because some
        // <clinit> methods add Utf8 references to our table.
        it = PrimordialClassLoader.loader.getAllTypes().iterator();
        while (it.hasNext()) {
            jq_Type t = (jq_Type)it.next();
            Reflection.getJDKType(t);
        }

        // get the set of compiled methods, because it is used during bootstrapping.
        CodeAllocator.getCompiledMethods();
        
        System.out.println("number of classes seen = "+PrimordialClassLoader.loader.getAllTypes().size());
        System.out.println("number of classes in image = "+objmap.boot_types.size());
        
        // During the process of initialization, etc., Utf8 objects could
        // have been allocated, causing the Utf8 table to grow.  We update
        // the static fields of the Utf8 class to catch these.
        // (Hopefully no other static fields have changed!?)
        jq_Class utf8_class = (jq_Class) PrimordialClassLoader.loader.getBSType("Ljoeq/UTF/Utf8;");
        jq_StaticField[] sfs = utf8_class.getDeclaredStaticFields();
        for (int j=0; j<sfs.length; ++j) {
            jq_StaticField sf = sfs[j];
            //System.out.println("Initializing static field: "+sf);
            objmap.initStaticField(sf);
        }
        System.out.println("Total number of Utf8 = "+(Utf8.size+1));

        // we shouldn't encounter any new Utf8 from this point
        Utf8.NO_NEW = true;

        // add all reachable members.
        System.out.println("Finding all reachable objects...");
        objmap.find_reachable(0);
        
        long traversaltime = System.currentTimeMillis() - starttime;

        // now that we have visited all reachable objects, jq.on_vm_startup is built
        int index = objmap.numOfEntries();
        HeapAddress addr = objmap.getOrAllocateObject(jq.on_vm_startup);
        Assert._assert(objmap.numOfEntries() > index);
        objmap.find_reachable(index);
        jq_StaticField _on_vm_startup = jq_class.getOrCreateStaticField("on_vm_startup", "Ljava/util/List;");
        jq_class.setStaticData(_on_vm_startup, addr);
        objmap.addDataReloc(_on_vm_startup.getAddress(), addr);

        // all done with traversal, no more objects can be added to the image.
        objmap.disableAllocations();
        
        System.out.println("Scanned: "+objmap.numOfEntries()+" objects, memory used: "+(Runtime.getRuntime().totalMemory()-Runtime.getRuntime().freeMemory())+"                    ");
        System.out.println("Scan time: "+traversaltime/1000f+"s");
        System.out.println("Image heap size = "+objmap.size());
        System.out.println("Image code size = "+bca.size());
        
        // update code min/max addresses
        // don't use initStaticField because it (re-)adds relocs
        objmap.initStaticField(CodeAllocator._lowAddress);
        objmap.initStaticField(CodeAllocator._highAddress);
        
        // dump it!
        FileOutputStream fos = new FileOutputStream(imageName);
        DirectBufferedFileOutputStream dbfos = new DirectBufferedFileOutputStream(fos);
        dbfos.order(ByteOrder.LITTLE_ENDIAN);
        starttime = System.currentTimeMillis();
        try {
            if (DUMP_COFF)
                objmap.dumpCOFF((ExtendedDataOutput) dbfos, rootm);
            else
                objmap.dumpELF((ExtendedDataOutput) dbfos, rootm);
        } finally {
            dbfos.close();
        }
        long dumptime = System.currentTimeMillis() - starttime;
        System.out.println("Dump time: "+dumptime/1000f+"s");
        
        if (false) {
            it = classset.iterator();
            while (it.hasNext()) {
                jq_Type t = (jq_Type)it.next();
                if (t == Unsafe._class) continue;
                Assert._assert(t.isClsInitialized());
                System.out.println(t+": "+objmap.getAddressOf(t).stringRep());
                if (t.isReferenceType()) {
                    jq_Reference r = (jq_Reference)t;
                    System.out.println("\tninterfaces "+r.getInterfaces().length+" vtable "+objmap.getAddressOf(r.getVTable()).stringRep());
                }
            }
        }
        
        System.out.println(rootm.getDefaultCompiledVersion());

        //objmap = null;
        //System.gc(); System.gc();
        //System.out.println("total memory = "+Runtime.getRuntime().totalMemory());
        //System.out.println("free memory = "+Runtime.getRuntime().freeMemory());
    }

    public static void dumpClassSet(Set s) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream("classlist"));
        for (Iterator i = s.iterator(); i.hasNext(); ) {
            jq_Type t = (jq_Type) i.next();
            dos.writeBytes(t + "\n");
        }
        dos.close();
    }
    
    public static void dumpMethodSet(Set s) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream("methodlist"));
        for (Iterator i = s.iterator(); i.hasNext(); ) {
            jq_Method t = (jq_Method) i.next();
            dos.writeBytes(t + "\n");
        }
        dos.close();
    }
    
    public static void err(String s) {
        System.err.println(s);
        System.exit(0);
    }

}

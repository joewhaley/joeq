/*
 * Bootstrapper.java
 *
 * Created on January 1, 2001, 11:46 PM
 *
 * @author  jwhaley
 * @version 
 */

package Main;

import jq;
import Allocator.*;
import ClassLib.ClassLibInterface;
import Clazz.*;
import Util.*;
import Run_Time.*;
import UTF.Utf8;
import Compil3r.BytecodeAnalysis.*;
import Compil3r.Reference.x86.*;
import Bootstrap.*;
import java.io.*;
import java.util.*;

import java.lang.reflect.Array;
import java.lang.reflect.Field;

public abstract class Bootstrapper implements ObjectLayout {

    private static BootImage objmap;
    
    public static void main(String[] args) throws IOException {
        
        String imageName = "jq.obj";
        //int startAddress = 0x00890000;
        String rootMethodClassName = "jq";
        String rootMethodName = "boot";
        String classList = null;
        String addToClassList = null;
        boolean TrimAllTypes = false;

        jq.Bootstrapping = true;
        
        String classpath = System.getProperty("java.class.path")+
                           System.getProperty("path.separator")+
                           System.getProperty("sun.boot.class.path");
        
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
            /*
            if (args[i].equals("-s")) { // start address
                startAddress = Integer.parseInt(args[++i], 16);
                ++i; continue;
            }
             */
            err("unknown command line argument: "+args[i]);
        }
        
        for (Iterator it = PrimordialClassLoader.classpaths(classpath); it.hasNext(); ) {
            String s = (String)it.next();
            PrimordialClassLoader.loader.addToClasspath(s);
        }
        
        Set nullStaticFields = ClassLibInterface.i.bootstrapNullStaticFields();
        Set nullInstanceFields = ClassLibInterface.i.bootstrapNullInstanceFields();
        System.out.println("Null static fields: "+nullStaticFields);
        System.out.println("Null instance fields: "+nullInstanceFields);

        // install bootstrap code allocator
        BootstrapCodeAllocator bca = new BootstrapCodeAllocator();
        DefaultCodeAllocator.default_allocator = bca;
        bca.init();
        
        // install object mapper
        ObjectTraverser obj_trav = new ObjectTraverser(nullStaticFields, nullInstanceFields);
        Reflection.obj_trav = obj_trav;
        Unsafe.installRemapper(objmap = new BootImage(obj_trav, bca));
        
        long starttime = System.currentTimeMillis();
        jq_Class c;
        c = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("L"+rootMethodClassName+";");
        c.load(); c.verify(); c.prepare();
        long loadtime = System.currentTimeMillis() - starttime;

        jq_StaticMethod rootm = null;
        Utf8 rootm_name = Utf8.get(rootMethodName);
        for(Iterator it = new ArrayIterator(c.getDeclaredStaticMethods());
            it.hasNext(); ) {
            jq_StaticMethod m = (jq_StaticMethod)it.next();
            if (m.getName() == rootm_name) {
                rootm = m;
                break;
            }
        }
        if (rootm == null)
            err("root method not found: "+rootMethodClassName+"."+rootMethodName);
        
        // initialize list of methods to invoke on startup
        jq.on_vm_startup = new LinkedList();
        
        Set classset = new HashSet();
        Set memberset;
        
        starttime = System.currentTimeMillis();
        if (addToClassList != null) {
            DataInputStream dis = new DataInputStream(new FileInputStream(addToClassList));
            for (;;) {
                String classname = dis.readLine();
                if (classname == null) break;
                if (classname.charAt(0) == '#') continue;
                jq_Type t = (jq_Type)PrimordialClassLoader.loader.getOrCreateBSType(classname);
                t.load(); t.verify(); t.prepare();
                classset.add(t);
            }
        }
        if (classList != null) {
            DataInputStream dis = new DataInputStream(new FileInputStream(classList));
            for (;;) {
                String classname = dis.readLine();
                if (classname == null) break;
                if (classname.charAt(0) == '#') continue;
                if (classname.endsWith("*")) {
                    jq.assert(classname.startsWith("L"));
                    Iterator i = PrimordialClassLoader.loader.listPackage(classname.substring(1, classname.length()-1));
                    while (i.hasNext()) {
                        String s = (String)i.next();
                        jq.assert(s.endsWith(".class"));
                        s = "L"+s.substring(0, s.length()-6)+";";
                        jq_Type t = (jq_Type)PrimordialClassLoader.loader.getOrCreateBSType(s);
                        t.load(); t.verify(); t.prepare();
                        classset.add(t);
                    }
                } else {
                    jq_Type t = (jq_Type)PrimordialClassLoader.loader.getOrCreateBSType(classname);
                    t.load(); t.verify(); t.prepare();
                    classset.add(t);
                }
            }
            memberset = new HashSet();
            Iterator i = classset.iterator();
            while (i.hasNext()) {
                jq_Type t = (jq_Type)i.next();
                if (t.isClassType()) {
                    jq_Class cl = (jq_Class)t;
                    jq_Method[] ms = cl.getDeclaredStaticMethods();
                    for (int k=0; k<ms.length; ++k) {
                        memberset.add(ms[k]);
                    }
                    ms = cl.getDeclaredInstanceMethods();
                    for (int k=0; k<ms.length; ++k) {
                        memberset.add(ms[k]);
                    }
                    ms = cl.getVirtualMethods();
                    for (int k=0; k<ms.length; ++k) {
                        memberset.add(ms[k]);
                    }
                }
            }
        } else {
            // traverse the code and data starting at the root set to find all necessary
            // classes and members.
            
            Trimmer trim = new Trimmer(rootm, obj_trav, !TrimAllTypes, classset);
            trim.go();

            System.out.println("Number of instantiated types: "+trim.getInstantiatedTypes().size());
            //System.out.println("Instantiated types: "+trim.getInstantiatedTypes());

            System.out.println("Number of necessary members: "+trim.getNecessaryMembers().size());
            //System.out.println("Necessary members: "+trim.getNecessaryMembers());

            // find all used classes.
            classset = trim.getNecessaryTypes();

            System.out.println("Number of necessary classes: "+classset.size());
            //System.out.println("Necessary classes: "+classset);

            if (TrimAllTypes) {
                // Trim all the types.
                Iterator it = classset.iterator();
                while (it.hasNext()) {
                    jq_Type t = (jq_Type)it.next();
                    System.out.println("Trimming type: "+t.getName());
                    jq.assert(t.isPrepared());
                    if (t.isClassType()) {
                        ((jq_Class)t).trim(trim);
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
            
            memberset = trim.getNecessaryMembers();
        }
        loadtime += System.currentTimeMillis() - starttime;
        System.out.println("Load time: "+loadtime);
        
        // initialize the set of boot types
        jq.boot_types = classset;
        
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
        
        // enable allocations
        objmap.enableAllocations();

        // allocate entrypoints first in bootimage.
        // NOTE: will only be first if java.lang.Object doesn't have any static members.
        SystemInterface._class.load();
        SystemInterface._class.verify();
        SystemInterface._class.prepare();
        SystemInterface._class.sf_initialize();

        //jq.assert(SystemInterface._entry.getAddress() == startAddress + ARRAY_HEADER_SIZE,
        //          "entrypoint is at "+jq.hex8(SystemInterface._entry.getAddress()));
        
        // initialize the static fields for all the necessary types
        starttime = System.currentTimeMillis();
        it = classset.iterator();
        while (it.hasNext()) {
            jq_Type t = (jq_Type)it.next();
            jq.assert(t.isPrepared());
            t.sf_initialize();
            // initialize static field values, too.
            if (t.isClassType()) {
                jq_Class k = (jq_Class)t;
                jq.assert((k.getSuperclass() == null) || classset.contains(k.getSuperclass()),
                          k.getSuperclass()+" (superclass of "+k+") is not in class set!");
                jq_StaticField[] sfs = k.getDeclaredStaticFields();
                for (int j=0; j<sfs.length; ++j) {
                    jq_StaticField sf = sfs[j];
                    objmap.initStaticField(sf);
                    objmap.addStaticFieldReloc(sf);
                }
            }
        }
        long sfinittime = System.currentTimeMillis() - starttime;
        System.out.println("SF init time: "+sfinittime);
        
        // turn off jq.Bootstrapping flag in image
        jq_Class jq_class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljq;");
        jq_class.setStaticData(jq_class.getOrCreateStaticField("Bootstrapping","Z"), 0);

        // compile versions of all necessary methods.
        starttime = System.currentTimeMillis();
        it = memberset.iterator();
        while (it.hasNext()) {
            jq_Member m = (jq_Member)it.next();
            if (m instanceof jq_Method) {
                jq_Method m2 = ((jq_Method)m);
                if (m2.getDeclaringClass() == Unsafe._class) continue;
                m2.compile();
            }
        }
        long compiletime = System.currentTimeMillis() - starttime;
        System.out.println("Compile time: "+compiletime);

        // initialize and add the jq_Class/jq_Array/jq_Primitive objects for all
        // necessary types.
        starttime = System.currentTimeMillis();
        it = classset.iterator();
        while (it.hasNext()) {
            jq_Type t = (jq_Type)it.next();
            jq.assert(t.isSFInitialized());
            
            if (t == Unsafe._class) continue;
            System.out.println("Compiling type: "+t.getName());
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

        //Object xxx = Assembler.x86.ExternalReference._heap_from;
	//Object yyy = ClassLib.sun13.java.io.Win32FileSystem._class;
        
        // get the set of compiled methods, because it is used during bootstrapping.
        CodeAllocator.getCompiledMethods();
        
        System.out.println("number of classes seen = "+PrimordialClassLoader.loader.getAllTypes().size());
        System.out.println("number of classes in image = "+jq.boot_types.size());
        
	// we shouldn't encounter any new Utf8 from this point
	Utf8.NO_NEW = true;

        // add all reachable members.
        System.out.println("Finding all reachable objects...");
        objmap.find_reachable(0);
        
        long traversaltime = System.currentTimeMillis() - starttime;
        System.out.println("Traversal time: "+traversaltime);

        // now that we have visited all reachable objects, jq.on_vm_startup is built
        int index = objmap.numOfEntries();
        int addr = objmap.getOrAllocateObject(jq.on_vm_startup);
        jq.assert(objmap.numOfEntries() > index);
        objmap.find_reachable(index);
        jq_StaticField _on_vm_startup = jq_class.getOrCreateStaticField("on_vm_startup", "Ljava/util/List;");
        jq_class.setStaticData(_on_vm_startup, addr);
        objmap.addDataReloc(_on_vm_startup.getAddress(), addr);

        // all done with traversal, no more objects can be added to the image.
        objmap.disableAllocations();
        
        System.out.println("number of objects = "+objmap.numOfEntries());
        System.out.println("heap size = "+objmap.size());
        System.out.println("code size = "+bca.size());
        
        // update code min/max addresses
        // don't use initStaticField because it (re-)adds relocs
        objmap.initStaticField(CodeAllocator._lowAddress);
        objmap.initStaticField(CodeAllocator._highAddress);
        
        // dump it!
        FileOutputStream fos = new FileOutputStream(imageName);
        starttime = System.currentTimeMillis();
        objmap.dumpCOFF(fos, rootm);
        long dumptime = System.currentTimeMillis() - starttime;
        System.out.println("Dump time: "+dumptime);
        
        it = classset.iterator();
        while (it.hasNext()) {
            jq_Type t = (jq_Type)it.next();
            if (t == Unsafe._class) continue;
            jq.assert(t.isClsInitialized());
            System.out.println(t+": "+jq.hex8(objmap.getAddressOf(t)));
            if (t.isReferenceType()) {
                jq_Reference r = (jq_Reference)t;
                System.out.println("\tninterfaces "+r.getInterfaces().length+" vtable "+jq.hex8(objmap.getAddressOf(r.getVTable())));
            }
        }
        
        System.out.println(rootm.getDefaultCompiledVersion());

        //objmap = null;
        //System.gc(); System.gc();
        //System.out.println("total memory = "+Runtime.getRuntime().totalMemory());
        //System.out.println("free memory = "+Runtime.getRuntime().freeMemory());
    }

    public static void err(String s) {
        System.err.println(s);
        System.exit(0);
    }

}

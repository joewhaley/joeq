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
import Clazz.*;
import Util.*;
import Run_Time.*;
import UTF.Utf8;
import Compil3r.Analysis.*;
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
        
        PrimordialClassLoader.loader.addToClasspath(classpath);
        
        Set nullStaticFields = new HashSet();
        nullStaticFields.add(Unsafe._remapper_object);
        nullStaticFields.add(ClassLib.sun13.java.lang.System._in);
        nullStaticFields.add(ClassLib.sun13.java.lang.System._out);
        nullStaticFields.add(ClassLib.sun13.java.lang.System._err);
        nullStaticFields.add(ClassLib.sun13.java.lang.System._props);
        nullStaticFields.add(Reflection._obj_trav);
        nullStaticFields.add(CodeAllocator._DEFAULT);
        nullStaticFields.add(ClassLib.sun13.java.lang.ClassLoader._loadedLibraryNames);
        nullStaticFields.add(ClassLib.sun13.java.lang.ClassLoader._systemNativeLibraries);
        nullStaticFields.add(ClassLib.sun13.java.lang.ClassLoader._nativeLibraryContext);
        System.out.println("Null static fields: "+nullStaticFields);

        // install bootstrap code allocator
        BootstrapCodeAllocator bca = new BootstrapCodeAllocator();
        CodeAllocator.DEFAULT = bca;
        bca.init();
        
        // install object mapper
        ObjectTraverser obj_trav = new ObjectTraverser(nullStaticFields);
        Reflection.obj_trav = obj_trav;
        Unsafe.installRemapper(objmap = new BootImage(obj_trav, bca));
        
        jq_Class c;
        c = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("L"+rootMethodClassName+";");
        c.load(); c.verify(); c.prepare();

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
        
        Set classset;
        Set memberset;
        
        if (classList != null) {
            DataInputStream dis = new DataInputStream(new FileInputStream(classList));
            classset = new HashSet();
            memberset = new HashSet();
            for (;;) {
                String classname = dis.readLine();
                if (classname == null) break;
                if (classname.charAt(0) == '#') continue;
                jq_Type t = (jq_Type)PrimordialClassLoader.loader.getOrCreateBSType(classname);
                t.load(); t.verify(); t.prepare();
                classset.add(t);
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
            Trimmer trim = new Trimmer(rootm, obj_trav, !TrimAllTypes);
            trim.go();

            System.out.println("Number of instantiated types: "+trim.getInstantiatedTypes().size());
            System.out.println("Instantiated types: "+trim.getInstantiatedTypes());

            System.out.println("Number of necessary members: "+trim.getNecessaryMembers().size());
            System.out.println("Necessary members: "+trim.getNecessaryMembers());

            // find all used classes.
            classset = trim.getNecessaryTypes();

            System.out.println("Number of necessary classes: "+classset.size());
            System.out.println("Necessary classes: "+classset);

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
            }
            
            memberset = trim.getNecessaryMembers();
        }
        
        // initialize the set of boot types
        jq.boot_types = classset;
        
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
        Iterator it = classset.iterator();
        while (it.hasNext()) {
            jq_Type t = (jq_Type)it.next();
            jq.assert(t.isPrepared());
            t.sf_initialize();
            // initialize static field values, too.
            if (t.isClassType()) {
                jq_Class k = (jq_Class)t;
                jq.assert((k.getSuperclass() == null) || classset.contains(k.getSuperclass()),
                          k.getSuperclass()+" is not in class set");
                jq_StaticField[] sfs = k.getDeclaredStaticFields();
                for (int j=0; j<sfs.length; ++j) {
                    jq_StaticField sf = sfs[j];
                    objmap.initStaticField(sf);
                    objmap.addStaticFieldReloc(sf);
                }
            }
        }
        // turn off jq.Bootstrapping flag in image
        jq_Class jq_class = (jq_Class)PrimordialClassLoader.loader.getOrCreateBSType("Ljq;");
        jq_class.setStaticData(jq_class.getOrCreateStaticField("Bootstrapping","Z"), 0);

        // compile versions of all necessary methods.
        it = memberset.iterator();
        while (it.hasNext()) {
            jq_Member m = (jq_Member)it.next();
            if (m instanceof jq_Method) {
                ((jq_Method)m).compile();
            }
        }

        // initialize and add the jq_Class/jq_Array/jq_Primitive objects for all
        // necessary types.
        it = classset.iterator();
        while (it.hasNext()) {
            jq_Type t = (jq_Type)it.next();
            jq.assert(t.isSFInitialized());
            if (t == Unsafe._class) continue;
            System.out.println("Compiling type: "+t.getName());
            t.cls_initialize();
            objmap.getOrAllocateObject(t);
        }
        
        // add all reachable members.
        System.out.println("Finding all reachable objects...");
        objmap.find_reachable();
        
        objmap.disableAllocations();
        
        // store entrypoint/trap handler addresses
        SystemInterface.entry_0 = rootm.getDefaultCompiledVersion().getEntrypoint();
        SystemInterface.trap_handler_8 = ExceptionDeliverer._trap_handler.getDefaultCompiledVersion().getEntrypoint();
        objmap.initStaticField(SystemInterface._entry);
        objmap.initStaticField(SystemInterface._trap_handler);
        
        // update code min/max addresses
        // don't use initStaticField because it (re-)adds relocs
        objmap.initStaticField(CodeAllocator._lowAddress);
        objmap.initStaticField(CodeAllocator._highAddress);
        
        // dump it!
        FileOutputStream fos = new FileOutputStream(imageName);
        objmap.dumpCOFF(fos);
        
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
    }

    public static void err(String s) {
        System.err.println(s);
        System.exit(0);
    }

}

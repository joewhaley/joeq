package Main;

import java.util.Iterator;

import Bootstrap.PrimordialClassLoader;
import ClassLib.ClassLibInterface;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Memory.StackAddress;
import Run_Time.Reflection;

public abstract class HostedVM {
    public static void initialize() {
        if (jq.RunningNative) return;
        
        jq.DontCompile = true;
        jq.boot_types = new java.util.HashSet();

        CodeAddress.FACTORY = new CodeAddress.CodeAddressFactory() {
            public int size() {
                return 4;
            }
            public CodeAddress getNull() {
                return null;
            }
        };
        HeapAddress.FACTORY = new HeapAddress.HeapAddressFactory() {
            public int size() {
                return 4;
            }
            
            public int logSize() {
                return 2;
            }
            
            public int pageAlign() {
                return 12; // 2**12 = 4096
            }

            public HeapAddress getNull() {
                return null;
            }

            public HeapAddress addressOf(Object o) {
                return null;
            }

            public HeapAddress address32(int val) {
                return null;
            }
        };
        StackAddress.FACTORY = new StackAddress.StackAddressFactory() {
            public int size() {
                return 4;
            }

            public StackAddress alloca(int a) {
                jq.UNREACHABLE();
                return null;
            }

            public StackAddress getBasePointer() {
                jq.UNREACHABLE();
                return null;
            }

            public StackAddress getStackPointer() {
                jq.UNREACHABLE();
                return null;
            }
        };
        String classpath = System.getProperty("sun.boot.class.path") + System.getProperty("path.separator") + System.getProperty("java.class.path");
        for (Iterator it = PrimordialClassLoader.classpaths(classpath); it.hasNext();) {
            String s = (String) it.next();
            PrimordialClassLoader.loader.addToClasspath(s);
        }
        Reflection.obj_trav = ClassLibInterface.DEFAULT.getObjectTraverser();
        Reflection.obj_trav.initialize();

	jq._crash = new SimpleCrash();
    }
}

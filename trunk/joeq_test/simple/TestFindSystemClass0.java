package simple;

import java.util.*;
import java.lang.reflect.*;
import joeq.Class.PrimordialClassLoader;
import joeq.Main.HostedVM;

public class TestFindSystemClass0 {
    public static void main(String[] args) throws Exception {
        Class c = Class.forName("java.lang.Package");
        Class[] a = new Class[] { java.lang.String.class };
        Method m = c.getDeclaredMethod("getSystemPackage0", a);
        m.setAccessible(true);
        Object r = m.invoke(null, (Object[])args);
        System.out.println(r);

        HostedVM.initialize();
        System.out.println(PrimordialClassLoader.loader.getPackagePath(args[0]));
    }
}

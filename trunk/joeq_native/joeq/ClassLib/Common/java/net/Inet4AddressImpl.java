// Inet4AddressImpl.java, created Fri Mar  7 11:01:56 2003 by joewhaley
// Copyright (C) 2001-3 John Whaley <jwhaley@alum.mit.edu>
// Licensed under the terms of the GNU LGPL; see COPYING for details.
package ClassLib.Common.java.net;

import Bootstrap.MethodInvocation;
import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_Method;
import Clazz.jq_NameAndDesc;
import Main.jq;
import Memory.Address;
import Memory.CodeAddress;
import Memory.HeapAddress;
import Run_Time.SystemInterface;
import Run_Time.Unsafe;
import Run_Time.SystemInterface.ExternalLink;
import Run_Time.SystemInterface.Library;
import Util.Assert;

/**
 * Inet4AddressImpl
 *
 * @author John Whaley <jwhaley@alum.mit.edu>
 * @version $Id$
 */
public class Inet4AddressImpl {
    
    static class hostent {
        String   h_name;      /* official name of host */
        String[] h_aliases;   /* alias list */
        int      h_addrtype;  /* host address type */
        int      h_length;    /* length of address */
        String[] h_addr_list; /* list of addresses */
    }
    
    public static hostent get_host_by_name(String name) {
        try {
            CodeAddress a = gethostbyname.resolve();
            byte[] b = SystemInterface.toCString(name);
            HeapAddress c = HeapAddress.addressOf(b);
            Unsafe.pushArgA(c);
            Unsafe.getThreadBlock().disableThreadSwitch();
            Address p = Unsafe.invokeA(a);
            Unsafe.getThreadBlock().enableThreadSwitch();
            hostent r = new hostent();
            r.h_name = SystemInterface.fromCString(p.peek());
            int count = 0;
            Address q = p.offset(HeapAddress.size()).peek();
            while (!q.peek().isNull()) {
                ++count;
                q = q.offset(HeapAddress.size());
            }
            r.h_aliases = new String[count];
            count = 0;
            q = p.offset(HeapAddress.size()).peek();
            while (!q.peek().isNull()) {
                r.h_aliases[count] = SystemInterface.fromCString(q.peek());
                ++count;
                q = q.offset(HeapAddress.size());
            }
            r.h_addrtype = p.offset(HeapAddress.size()*2).peek4();
            r.h_length = p.offset(HeapAddress.size()*2+4).peek4();
            count = 0;
            q = p.offset(HeapAddress.size()*2+8).peek();
            while (!q.peek().isNull()) {
                ++count;
                q = q.offset(HeapAddress.size());
            }
            count = 0;
            q = p.offset(HeapAddress.size()*2+8).peek();
            while (!q.peek().isNull()) {
                r.h_addr_list[count] = SystemInterface.fromCString(q.peek());
                ++count;
                q = q.offset(HeapAddress.size());
            }
            return r;
        } catch (Throwable x) { Assert.UNREACHABLE(); }
        return null;
    }
    
    public static /*final*/ ExternalLink gethostbyname;

    static {
        if (jq.RunningNative) boot();
        else if (jq.on_vm_startup != null) {
            jq_Class c = (jq_Class) PrimordialClassLoader.loader.getOrCreateBSType("Ljava/net/Inet4AddressImpl;");
            jq_Method m = c.getDeclaredStaticMethod(new jq_NameAndDesc("boot", "()V"));
            MethodInvocation mi = new MethodInvocation(m, null);
            jq.on_vm_startup.add(mi);
        }
    }

    public static void boot() {
        Library winsock = SystemInterface.registerLibrary("ws2_32");

        if (winsock != null) {
            gethostbyname = winsock.resolve("gethostbyname");
        } else {
            gethostbyname = null;
        }

    }

    public java.lang.String getLocalHostName() throws java.net.UnknownHostException {
        return null;
    }
    public byte[][] lookupAllHostAddr(java.lang.String hostname) throws java.net.UnknownHostException {
        return null;
    }
    public java.lang.String getHostByAddr(byte[] addr) throws java.net.UnknownHostException {
        return null;
    }
}

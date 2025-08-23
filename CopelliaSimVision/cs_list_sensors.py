# cs_debug_connection.py
import sys, time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

PORTS = [23000, 23001, 23002, 23003]

def try_port(port):
    try:
        client = RemoteAPIClient('127.0.0.1', port=port, debug=False)
        sim = client.getObject('sim')
        # ensure scene initialized
        if sim.getSimulationState() == sim.simulation_stopped:
            # no need to start; just count objects at rest
            pass
        hs = sim.getObjectsInTree(sim.handle_scene, 0, 0)
        names = []
        for h in hs[:40]:
            nm = ""
            try:
                nm = sim.getObjectAlias(h, 1)
            except Exception:
                try:
                    nm = sim.getObjectName(h)
                except Exception:
                    nm = ""
            names.append(nm)
        return len(hs), names
    except Exception as e:
        return None, [str(e)]

def main():
    print("Probing CoppeliaSim ZMQ Remote API servers on localhost...")
    best = None
    for p in PORTS:
        cnt, names = try_port(p)
        if cnt is None:
            print(f" - port {p}: cannot connect ({names[0]})")
        else:
            print(f" - port {p}: {cnt} objects")
            if cnt > 0:
                print("   sample objects:")
                for nm in names[:10]:
                    print("    ·", nm)
            if best is None or (cnt or 0) > (best[1] or 0):
                best = (p, cnt)
    if best and best[1] and best[1] > 0:
        print(f"\n✅ Use port {best[0]} (has {best[1]} objects).")
    else:
        print("\n⚠️ No scene with objects found. Open your scene in CoppeliaSim, then ensure the ZMQ server is running (Add-ons → ZMQ remote API → Start) and note the port.")

if __name__ == "__main__":
    main()

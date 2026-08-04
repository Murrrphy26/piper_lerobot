for i in {0..5}; do
    if [ -e /dev/video$i ]; then
        echo -n "/dev/video$i -> "
        udevadm info -q property -n /dev/video$i | grep ID_PATH | cut -d= -f2
    fi
done
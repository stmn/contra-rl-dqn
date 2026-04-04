-- FCEUX Lua script: sends RAM + full-res RGB screen to Python

local ACTION_FILE = "/tmp/contra_action.txt"
local STATE_FILE = "/tmp/contra_state.bin"

local ACTION_MAP = {
    [0]  = {},
    [1]  = {right=true},
    [2]  = {left=true},
    [3]  = {up=true},
    [4]  = {down=true},
    [5]  = {A=true},
    [6]  = {B=true},
    [7]  = {right=true, A=true},
    [8]  = {right=true, B=true},
    [9]  = {right=true, A=true, B=true},
    [10] = {left=true, A=true},
    [11] = {left=true, B=true},
    [12] = {up=true, B=true},
    [13] = {down=true, B=true},
    [14] = {right=true, up=true, B=true},
    [15] = {right=true, down=true, B=true},
}

local action = 0
local frame = 0
local FRAME_SKIP = 2
local SCREEN_W = 256
local SCREEN_H = 224

os.remove(ACTION_FILE)
os.remove(STATE_FILE)

emu.message("Agent bridge ready — select 1 Player")

while true do
    frame = frame + 1

    if frame % FRAME_SKIP == 0 then
        -- Write binary: 2048 bytes RAM + 256*224*3 bytes RGB screen
        local sf = io.open(STATE_FILE .. ".tmp", "wb")
        if sf then
            -- RAM (2048 bytes)
            for addr = 0, 2047 do
                sf:write(string.char(memory.readbyte(addr)))
            end
            -- Screen RGB (256*224*3 = 172032 bytes)
            for y = 0, SCREEN_H - 1 do
                for x = 0, SCREEN_W - 1 do
                    local r, g, b = emu.getscreenpixel(x, y, false)
                    sf:write(string.char(r, g, b))
                end
            end
            sf:close()
            os.rename(STATE_FILE .. ".tmp", STATE_FILE)
        end

        -- Read action (non-blocking)
        local af = io.open(ACTION_FILE, "r")
        if af then
            local line = af:read("*l")
            af:close()
            if line then
                local a = tonumber(line)
                if a and a >= 0 and a <= 15 then
                    action = a
                end
            end
        end
    end

    joypad.set(1, ACTION_MAP[action] or {})

    -- Show $580-$58F values on screen
    local line = ""
    for slot = 0, 15 do
        local val = memory.readbyte(0x580 + slot)
        if val > 0 then
            line = line .. slot .. ":" .. val .. " "
        end
    end
    if line ~= "" then
        gui.text(2, 8, "$580: " .. line, "yellow", "black")
    end

    emu.frameadvance()
end
